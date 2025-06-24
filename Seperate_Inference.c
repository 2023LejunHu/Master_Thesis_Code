#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdint.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#ifdef OMP
#include <omp.h>
#endif
// our own utilities
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "llmc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "llmc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
#include "llmc/dataloader.h"

// ----------------------------------------------------------------------------
// all the individual layers' forward and backward passes

// B = batch_size            // number of samples in a batch
// T = sequence_length       // length of the input token sequence
// C = channels              // number of channels (i.e., embedding dimension)
// V = vocab_size            // size of the vocabulary
// Vp = padded_vocab_size    // padded vocabulary size (aligned to multiple of 128)
// NH = num_heads            // number of attention heads
// L = num_layers            // number of Transformer layers
// OC = output_channels      // number of output channels (matmul output dimension)
// C3 = 3 * C                // concatenated size of Query, Key, and Value vectors
// hs = C / NH               // head size, dimension per attention head
// eps = 1e-5                // small epsilon used for numerical stability in LayerNorm
// LOOP_UNROLL = 8           // loop unrolling factor in matmul
// GELU_SCALING_FACTOR = sqrtf(2.0f / M_PI)  // scaling factor for GELU activation
// maxT = max_seq_len        // maximum sequence length
// ix = token_index          // index in the vocabulary

/////////////////////////////////
/////////////////////////////////

// GPT-2 model configuration reference:
// max_seq_len: 1024
// vocab_size: 50257
// padded_vocab_size: 50304
// num_layers: 12
// num_heads: 12
// channels: 768
// num_parameters: 124,475,904
// num_activations: 73,347,840


#define NUM_PARAMETER_TENSORS 16  // number of parameter tensors
#define NUM_ACTIVATION_TENSORS 23 // number of activation tensors

// Data structure to store activation values of GPT-2 during forward pass,
// including intermediate results, normalization values, attention outputs,
// and final predicted probabilities.
// These activations are essential for both inference and training,
// especially for computing gradients in backward pass.

typedef struct
{
    float *encoded;   // (B, T, C) Output of the embedding layer, also used as input to transformer blocks
    float *ln1;       // (L, B, T, C) Output of LayerNorm1
    float *ln1_mean;  // (L, B, T) Mean of LayerNorm1, used for gradient computation
    float *ln1_rstd;  // (L, B, T) Inverse std of LayerNorm1, used for gradient computation
    float *qkv;       // (L, B, T, 3*C) Concatenated Query, Key and Value matrices
    float *atty;      // (L, B, T, C) Attention output: context vectors for each token
    float *preatt;    // (L, B, NH, T, T) Raw attention scores (Q·K^T), before softmax; NH is number of heads
    float *att;       // (L, B, NH, T, T) Attention weights after softmax
    float *attproj;   // (L, B, T, C) Projected attention output
    float *residual2; // (L, B, T, C) Output after second residual connection
    float *ln2;       // (L, B, T, C)
    float *ln2_mean;  // (L, B, T)
    float *ln2_rstd;  // (L, B, T)
    float *fch;       // (L, B, T, 4*C) Output of first linear layer in MLP
    float *fch_gelu;  // (L, B, T, 4*C) Output of GeLU activation
    float *fcproj;    // (L, B, T, C) Output of second linear projection in MLP
    float *residual3; // (L, B, T, C)
    float *lnf;       // (B, T, C) Final LayerNorm output
    float *lnf_mean;  // (B, T) Final LayerNorm mean
    float *lnf_rstd;  // (B, T) Final LayerNorm inverse std
    float *logits;    // (B, T, V) Final model output logits, before softmax
    float *probs;     // (B, T, V) Predicted probabilities after softmax
    float *losses;    // (B, T) Cross-entropy loss per position
} ActivationTensors;

typedef struct
{
    int max_seq_len;       // max sequence length, e.g. 1024
    int vocab_size;        // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers;        // number of layers, e.g. 12
    int num_heads;         // number of heads in attention, e.g. 12
    int channels;          // number of channels, e.g. 768
} GPT2Config;

// the parameters of the model
typedef struct
{
    float *wte;      // (V, C) Word Token Embeddings, where V is vocab size and C is embedding dimension
    float *wpe;      // (maxT, C) Word Positional Embeddings
    float *ln1w;     // (L, C) LayerNorm1 weights
    float *ln1b;     // (L, C) LayerNorm1 biases
    float *qkvw;     // (L, 3*C, C) Weights for Query, Key, and Value
    float *qkvb;     // (L, 3*C) Biases for Query, Key, and Value
    float *attprojw; // (L, C, C) Attention projection weights
    float *attprojb; // (L, C) Attention projection biases
    float *ln2w;     // (L, C) LayerNorm2 weights
    float *ln2b;     // (L, C) LayerNorm2 biases
    float *fcw;      // (L, 4*C, C) MLP FC1 weights (C → 4C)
    float *fcb;      // (L, 4*C) MLP FC1 biases
    float *fcprojw;  // (L, C, 4*C) MLP FC2 weights (4C → C)
    float *fcprojb;  // (L, C) MLP FC2 biases
    float *lnfw;     // (C) Final LayerNorm weights
    float *lnfb;     // (C) Final LayerNorm biases
} ParameterTensors;

typedef struct
{
    ParameterTensors params;     // Parameters stored on device
    float *device_params_memory; // Starting address of all parameter memory
    ActivationTensors acts;      // Activations stored on device
    float *device_acts_memory;   // Starting address of all activation memory
    int allocated_channels;      // Number of channels allocated to this device
} gpt_device;

typedef struct
{
    GPT2Config config;           // Model configuration
    gpt_device devices[3];       // Fixed to 3 devices; the number must divide 768 (channels) and be a multiple of 12 (num_heads)
    size_t param_sizes[NUM_PARAMETER_TENSORS];
    size_t act_sizes[NUM_ACTIVATION_TENSORS];
    size_t num_parameters;       // Total number of parameters
    size_t num_activations;      // Total number of activations
    ParameterTensors all_params; // Storage for all parameters
    float *all_params_memory;
    ActivationTensors all_acts;  // Storage for all activations
    float *all_acts_memory;
    int batch_size;              // Current forward pass batch size (B)
    int seq_len;                 // Current forward pass sequence length (T)
    int *inputs;                 // Input token indices for the current forward pass
} GPT2_Sep;


void fill_in_parameter_sizes(size_t *param_sizes, GPT2Config config)
{
    size_t Vp = config.padded_vocab_size;
    size_t C = config.channels;
    size_t maxT = config.max_seq_len;
    size_t L = config.num_layers;
    param_sizes[0] = Vp * C;           // wte weights token embeddings
    param_sizes[1] = maxT * C;         // wpe weights positional embeddings
    param_sizes[2] = L * C;            // ln1w    Layer Normalization 1 weights
    param_sizes[3] = L * C;            // ln1b    Layer Normalization 1 biases
    param_sizes[4] = L * (3 * C) * C;  // qkvw  Query, Key, Value Weights
    param_sizes[5] = L * (3 * C);      // qkvb  Query, Key, Value Biases
    param_sizes[6] = L * C * C;        // attprojw  Attention Projection Weights
    param_sizes[7] = L * C;            // attprojb  Attention Projection Biases
    param_sizes[8] = L * C;            // ln2w  Layer Normalization 2 weights
    param_sizes[9] = L * C;            // ln2b  Layer Normalization 2 biases
    param_sizes[10] = L * (4 * C) * C; // fcw  Feedforward Weights
    param_sizes[11] = L * (4 * C);     // fcb         Feedforward Biases
    param_sizes[12] = L * C * (4 * C); // fcprojw Feedforward Projection Weights
    param_sizes[13] = L * C;           // fcprojb Feedforward Projection Biases
    param_sizes[14] = C;               // lnfw  LayerNorm Final Weights
    param_sizes[15] = C;               // lnfb  LayerNorm Final Biases
}

// allocate memory for the parameters and point the individual tensors to the right places
float *malloc_and_point_parameters(ParameterTensors *params, size_t *param_sizes)
{
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        num_parameters += param_sizes[i];
    }
    // malloc all parameters all at once
    float *params_memory = (float *)mallocCheck(num_parameters * sizeof(float));
    // assign all the tensors
    float **ptrs[] = {
        &params->wte, &params->wpe, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb};
    float *params_memory_iterator = params_memory;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        *(ptrs[i]) = params_memory_iterator;
        params_memory_iterator += param_sizes[i];
    }
    return params_memory;
}

// load_factors is an array of 3 floats, each representing the percentage of channels allocated to each device
void gpt2_build_from_checkpoint_sep(GPT2_Sep *model, const char *checkpoint_path, float load_factors[3])
{
    FILE *model_file = fopenCheck(checkpoint_path, "rb"); // read binary
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);

    if (model_header[0] != 20240326)
    {
        printf("Bad magic model file\n");
        exit(1);
    }
    if (model_header[1] != 3)
    {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // read in hyperparameters
    size_t maxT, V, Vp, L, NH, C; // size_t to prevent int overflow
    model->config.max_seq_len = maxT = model_header[2];
    model->config.vocab_size = V = model_header[3];
    model->config.num_layers = L = model_header[4];
    model->config.num_heads = NH = model_header[5];
    model->config.channels = C = model_header[6];
    model->config.padded_vocab_size = Vp = model_header[7];
    printf("[GPT-2-Seperate]\n");
    printf("max_seq_len: %zu\n", maxT);
    printf("vocab_size: %zu\n", V);
    printf("padded_vocab_size: %zu\n", Vp);
    printf("num_layers: %zu\n", L);
    printf("num_heads: %zu\n", NH);
    printf("channels: %zu\n", C);

    // fill in parameter sizes
    fill_in_parameter_sizes(model->param_sizes, model->config);
    // count the number of parameters
    size_t num_parameters = 0;
    for (size_t i = 0; i < NUM_PARAMETER_TENSORS; i++)
    {
        num_parameters += model->param_sizes[i];
    }
    printf("num_parameters: %zu\n", num_parameters);
    model->num_parameters = num_parameters; // record the number of parameters in the model

    // it starts to allocate memory for all the parameters and read them in
    float total_weight = 0;
    float weight_ratios[3];
    for (int i = 0; i < 3; i++)
    {
        total_weight += (100 - load_factors[i]);
    }
    for (int i = 0; i < 3; i++)
    {
        weight_ratios[i] = (100 - load_factors[i]) / total_weight;
    }
    int total_channels = 768; 
    int min_unit = 64;        // the minimum unit of channel allocation, must be a multiple of 12 (num_heads) and divide 768 evenly
    int total_alloc = 0;
    // alloc is an array to store the allocated channels for each device
    int alloc[3] = {0};

    // 1. calculate initial allocation based on weight ratios
    for (int i = 0; i < 3; i++)
    {
        alloc[i] = round((total_channels * weight_ratios[i]) / min_unit) * min_unit;
        total_alloc += alloc[i];
    }

    // 2. refine allocation to ensure total channels is 768
    int remaining = total_channels - total_alloc;

    while (remaining != 0)
    {
        for (int i = 0; i < 3; i++)
        {
            if (remaining == 0)
                break;

            // calculate the adjustment for this device
            int adjust = (remaining > 0) ? min_unit : -min_unit;

            // prevent over-allocation or under-allocation
            if (alloc[i] + adjust >= 0)
            {
                alloc[i] += adjust;
                remaining -= adjust;
            }
        }
    }

    // 3. print the allocation results
    for (int i = 0; i < 3; i++)
    {
        printf("Device %d allocated %d channels\n", i, alloc[i]);
    }

    // it starts to allocate memory for all the parameters and read them in

    // read all the parameters from file
    model->all_params_memory = malloc_and_point_parameters(&model->all_params, model->param_sizes);
    freadCheck(model->all_params_memory, sizeof(float), num_parameters, model_file);
    fcloseCheck(model_file);

    // allocate space for all the activations if needed
    size_t offset = 0;
    for (int i = 0; i < 3; i++)
    {
        size_t device_deviation = 0;
        for (int n = 0; n < i; n++)
        {
            device_deviation += alloc[n];
        }
        offset = 0;
        // printf("Device %d allocated %d channels\n", i, alloc[i]);
        model->devices[i].allocated_channels = alloc[i];
        // printf("Device %d, allocated %d channels\n", i, alloc[i]);

        // calculate the total size of parameters for this device
        size_t device_param_size = 0;
        for (size_t j = 0; j < NUM_PARAMETER_TENSORS; j++)
        {

            device_param_size += (size_t)(model->param_sizes[j] * ((float)alloc[i] / total_channels));
        }
        // printf("Device Number: %d, device_param_size: %zu\n", i, device_param_size);
        //  The device is allocated a complete piece of memory.
        model->devices[i].device_params_memory = (float *)mallocCheck(device_param_size * sizeof(float));
        // printf("Device Number: %d, malloc success\n",i);
        // set the device's params to point to the allocated memory
        float **ptrs[] = {
            &model->devices[i].params.wte, &model->devices[i].params.wpe,
            &model->devices[i].params.ln1w, &model->devices[i].params.ln1b,
            &model->devices[i].params.qkvw, &model->devices[i].params.qkvb,
            &model->devices[i].params.attprojw, &model->devices[i].params.attprojb,
            &model->devices[i].params.ln2w, &model->devices[i].params.ln2b,
            &model->devices[i].params.fcw, &model->devices[i].params.fcb,
            &model->devices[i].params.fcprojw, &model->devices[i].params.fcprojb,
            &model->devices[i].params.lnfw, &model->devices[i].params.lnfb};

        float *device_params_iterator = model->devices[i].device_params_memory;
        float *src_params_iterator = model->all_params_memory;

        // printf("Device Number: %d, stage 2 success\n",i);

        for (size_t j = 0; j < NUM_PARAMETER_TENSORS; j++)
        {
            size_t tensor_size = model->param_sizes[j];
            size_t device_tensor_size = 0;

            device_tensor_size = (size_t)(tensor_size * ((float)alloc[i] / total_channels));
            *(ptrs[j]) = device_params_iterator;

            // copy the parameters for this tensor
            size_t row_size = total_channels;         // channels in one row
            size_t device_row_size = alloc[i];        // channels allocated to this device
            size_t num_rows = tensor_size / row_size; // calculate the number of rows in this tensor

            // printf("number of row of device %d in parameter%ld: %ld\n",i,j,num_rows);

            for (size_t r = 0; r < num_rows; r++)
            {
                // if(i==1){
                //     printf("Device Number: %d, allocating row %ld\n",i,r);
                // }
                memcpy(device_params_iterator + r * device_row_size,
                       src_params_iterator + offset + r * row_size + device_deviation,
                       device_row_size * sizeof(float));
            }

            offset += model->param_sizes[j]; // set offset for the next tensor
            device_params_iterator += device_tensor_size;
        }
        // printf("Device Number: %d, stage 3 success\n",i);
    }
    model->all_acts_memory = NULL;
    model->batch_size = 0; // the batch size (B) of current forward pass
    model->seq_len = 0;

    for (int i = 0; i < 3; i++)
    {
        model->devices[i].device_acts_memory = NULL;
    }
    model->inputs = NULL;

    ///////////////////////////////
    // start checking parameters
    /////////////////////////////
    printf("start checking qkvw...\n");

    for (int l = 0; l < L; l++)
    {
        for (int out_c = 0; out_c < 3 * C; out_c++)
        { // output dimension
            for (int in_c = 0; in_c < C; in_c++)
            { //input dimension
                float expected = model->all_params.qkvw[l * 3 * C * C + out_c * C + in_c];

                // find which device this out_c belongs to
                int device_id = -1;
                int channel_base = 0;
                for (int d = 0; d < 3; d++)
                {
                    int C_dev = model->devices[d].allocated_channels;
                    if (in_c < channel_base + C_dev)
                    {
                        device_id = d;
                        break;
                    }
                    channel_base += C_dev;
                }
                if (device_id == -1)
                {
                    printf("erro, cannot match output channel %d to any device\n", out_c);
                    exit(1);
                }

                int local_in_c = in_c - channel_base;
                size_t C_dev = model->devices[device_id].allocated_channels;

                float actual = model->devices[device_id].params.qkvw[l * 3 * C_dev * C + out_c * C_dev + local_in_c];

                if (fabs(expected - actual) != 0)
                {
                    printf(" mismatch @ layer %d, out_c %d, in_c %d → device %d (local_out_c %d)\n", l, out_c, in_c, device_id, local_in_c);
                    printf("expected: %.6f, actual: %.6f\n", expected, actual);
                    exit(1);
                }
            }
        }
    }
    printf("qkvw check passed！\n");

    printf("start checking qkvb ...\n");
    for (int l = 0; l < L; l++)
    {
        for (int out_c = 0; out_c < 3 * C; out_c++)
        { // output dimension
            float expected = model->all_params.qkvb[l * 3 * C + out_c];

            // find which device this out_c belongs to
            int device_id = -1;
            int channel_base = 0;

            int remainder = out_c % C; // col
            int quotient = out_c / C; //row

            for (int d = 0; d < 3; d++)
            {
                int C_dev = model->devices[d].allocated_channels;
                if (remainder < channel_base + C_dev)
                {
                    device_id = d;
                    break;
                }
                channel_base += C_dev;
            }
            if (device_id == -1)
            {
                printf("erro, cannot match output channel %d to any device\n", out_c);
                exit(1);
            }

            int local_out_c = remainder - channel_base;
            size_t C_dev = model->devices[device_id].allocated_channels;

            float actual = model->devices[device_id].params.qkvb[l * 3 * C_dev + quotient * C_dev + local_out_c];

            if ((expected - actual) != 0)
            {
                printf(" mismatch @ layer %d, out_c %d → device %d (local_out_c %d)\n", l, out_c, device_id, local_out_c);
                printf("expected: %.6f, actual: %.6f\n", expected, actual);
                exit(1);
            }
        }
    }
    printf("qkvb check passed\n");
}

void encoder_forward_sep(GPT2_Sep *model, int *inputs, size_t B, size_t T)
{
    // run across all devices
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels; // number of channels assigned to this device
        float *encoded = dev->acts.encoded;  // output tensor on this device
        float *wte = dev->params.wte;        // token embedding table
        float *wpe = dev->params.wpe;        // positional embedding table

        // iterate over the batch and time dimensions
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                // compute the output location for current batch and time step
                float *out_bt = encoded + (b * T + t) * C_dev;

                // get token ID
                int ix = inputs[b * T + t];

                // get the starting addresses of token and position embeddings
                float *wte_ix = wte + ix * C_dev; // token embedding for the current token (partial C_dev dimension)
                float *wpe_t = wpe + t * C_dev;   // positional embedding for timestep t (partial C_dev dimension)

                // compute final embedding vector by summing token and position embeddings
                for (int c = 0; c < C_dev; c++)
                {
                    out_bt[c] = wte_ix[c] + wpe_t[c];
                }
            }
        }
    }
}


void print_encoded_sep(GPT2_Sep *model, int b, int t)
{
    int B = model->batch_size;
    int T = model->seq_len;
    int C = model->config.channels;

    printf("\n[SEP] encoded @ (b=%d, t=%d):\n", b, t);

    int printed = 0;
    while (printed < C)
    {
        printf("channels [%d]: ", printed);

        // find which device this b, t belongs to
        int device_id = -1;
        int channel_base = 0;
        for (int d = 0; d < 3; d++)
        {
            int C_dev = model->devices[d].allocated_channels;
            if (printed < channel_base + C_dev)
            {
                device_id = d;
                break;
            }
            channel_base += C_dev;
        }

        int local_c = printed - channel_base;
        gpt_device *dev = &model->devices[device_id];
        float *encoded = dev->acts.encoded;
        int C_dev = dev->allocated_channels;

        float val = encoded[(b * T + t) * C_dev + local_c];
        printf("%.4f ", val);

        printf("\n");
        printed += 30;
    }
}

void layernorm_sep_1(float *mean_local, float *inp, int B, int T, int C_dev)
{
    // calculate local mean for each (b, t) pair
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            float *x = inp + (b * T + t) * C_dev;
            float m = 0.0f;

            // accumlate the sum of the input vector
            for (int i = 0; i < C_dev; i++)
            {
                m += x[i];
            }
            mean_local[b * T + t] = m; // local mean for (b, t) pair, aggregate later
        }
    }
}

void layernorm_sep_2(float *var_local, float *inp, float *mean_global, int B, int T, int C_dev)
{
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            float *x = inp + (b * T + t) * C_dev;
            float mean = mean_global[b * T]; 
            double v = 0.0f;

            // calculate local variance for each (b, t) pair
            for (int i = 0; i < C_dev; i++)
            {
                double xshift = x[i] - mean;
                v += xshift * xshift;
            }
            var_local[b * T + t] = v; // local variance for (b, t) pair, aggregate later
        }
    }
}

void layernorm_sep_3(float *out, float *inp, float *mean_global, double *rstd_global,
                     float *weight, float *bias, int B, int T, int C_dev)
{
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            float *x = inp + (b * T + t) * C_dev;
            float *out_bt = out + (b * T + t) * C_dev;
            float mean = mean_global[b * T];
            double rstd = rstd_global[b * T]; // reciprocal of standard deviation for (b, t) pair

            // normalize and apply linear transformation
            for (int i = 0; i < C_dev; i++)
            {
                double n = (x[i] - mean) * rstd;            // normalization
                out_bt[i] = (float)n * weight[i] + bias[i]; // linear transformation
            }
        }
    }
}

// call this function to perform LayerNorm forward pass for a single layer
void layernorm_forward_sep_1(GPT2_Sep *model, int l, int B, int T)
{
    // Step 1: Each device computes its local mean
    float mean_global[B * T]; // global mean across all devices
    memset(mean_global, 0, sizeof(mean_global));

    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        // void layernorm_sep_1(float *mean_local, float *inp, int B, int T, int C_dev)
        layernorm_sep_1(dev->acts.ln1_mean + l * B * T,
                        (l == 0) ? dev->acts.encoded : dev->acts.residual3 + (l - 1) * B * T * dev->allocated_channels,
                        B, T, dev->allocated_channels);
    }

    // Step 2: Aggregate global mean across all devices
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int i = 0; i < 3; i++)
            {
                mean_global[b * T + t] += model->devices[i].acts.ln1_mean[l * B * T + b * T + t];
            }
            mean_global[b * T + t] /= model->config.channels; // normalize the global mean
        }
    }

    // Debug print: global mean
    // if(l == 0){
    //     for (int i = 0; i < B * T; i++)
    //     {
    //         printf("SEP mean_global[%d]: %.4f\n", i, mean_global[i]);
    //     }
    // }

    // Step 3: Each device computes its local variance
    double var_global[B * T]; // global variance across all devices
    memset(var_global, 0, sizeof(var_global));

    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        layernorm_sep_2(dev->acts.ln1_rstd + l * B * T,
                        (l == 0) ? dev->acts.encoded : dev->acts.residual3 + (l - 1) * B * T * dev->allocated_channels,
                        mean_global, B, T, dev->allocated_channels); // temporarily store variance in rstd
    }

    // Step 4: Compute inverse standard deviation from global variance
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int i = 0; i < 3; i++)
            {
                var_global[b * T + t] += model->devices[i].acts.ln1_rstd[l * B * T + b * T + t];
            }
            var_global[b * T + t] /= model->config.channels;
            var_global[b * T + t] = 1.0f / sqrtf(var_global[b * T + t] + 1e-5f); // final inverse std
        }
    }

    // Debug print: global inverse std
    if (l == 0)
    {
        for (int i = 0; i < B * T; i++)
        {
            if (i % 10 == 0)
            {
                printf("SEP rstd_global[%d]: %.4f\n", i, var_global[i]);
            }
        }
    }

    // Step 5: Each device performs final normalization
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        layernorm_sep_3(dev->acts.ln1 + l * B * T * dev->allocated_channels,
                        (l == 0) ? dev->acts.encoded : dev->acts.residual3 + (l - 1) * B * T * dev->allocated_channels,
                        mean_global, var_global,
                        dev->params.ln1w + l * dev->allocated_channels,
                        dev->params.ln1b + l * dev->allocated_channels,
                        B, T, dev->allocated_channels);
    }
}


void layernorm_forward_sep_1_modified(GPT2_Sep *model, int l, int B, int T)
{
    int C = model->config.channels;

    float mean_global[B * T];
    float rstd_global[B * T];

    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {

            float sum = 0.0f;
            for (int c = 0; c < C; c++)
            {
                int device_id = -1;
                int channel_base = 0;
                for (int d = 0; d < 3; d++)
                {
                    int C_dev = model->devices[d].allocated_channels;
                    if (c < channel_base + C_dev)
                    {
                        device_id = d;
                        break;
                    }
                    channel_base += C_dev;
                }

                int local_c = c - channel_base;
                gpt_device *dev = &model->devices[device_id];
                float *x = (l == 0) ? dev->acts.encoded : dev->acts.residual3 + (l - 1) * B * T * dev->allocated_channels;
                float val = x[(b * T + t) * dev->allocated_channels + local_c];
                sum += val;
            }

            float mean = sum / C;
            mean_global[b * T + t] = mean;

            float var = 0.0f;
            for (int c = 0; c < C; c++)
            {
                int device_id = -1;
                int channel_base = 0;
                for (int d = 0; d < 3; d++)
                {
                    int C_dev = model->devices[d].allocated_channels;
                    if (c < channel_base + C_dev)
                    {
                        device_id = d;
                        break;
                    }
                    channel_base += C_dev;
                }

                int local_c = c - channel_base;
                gpt_device *dev = &model->devices[device_id];
                float *x = (l == 0) ? dev->acts.encoded : dev->acts.residual3 + (l - 1) * B * T * dev->allocated_channels;
                float val = x[(b * T + t) * dev->allocated_channels + local_c];
                float diff = val - mean;
                var += diff * diff;
            }

            rstd_global[b * T + t] = 1.0f / sqrtf(var / C + 1e-5f);
        }
    }

    for (int d = 0; d < 3; d++)
    {
        gpt_device *dev = &model->devices[d];
        int C_dev = dev->allocated_channels;
        float *inp = (l == 0) ? dev->acts.encoded : dev->acts.residual3 + (l - 1) * B * T * dev->allocated_channels;
        float *out = dev->acts.ln1 + l * B * T * C_dev;
        float *weight = dev->params.ln1w + l * C_dev;
        float *bias = dev->params.ln1b + l * C_dev;

        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float mean = mean_global[b * T + t];
                float rstd = rstd_global[b * T + t];

                float *x = inp + (b * T + t) * C_dev;
                float *y = out + (b * T + t) * C_dev;

                for (int i = 0; i < C_dev; i++)
                {
                    float norm = (x[i] - mean) * rstd;
                    y[i] = norm * weight[i] + bias[i];
                }
            }
        }
    }
}

void print_ln1_sep(GPT2_Sep *model, int l, int b, int t)
{

    int B = model->batch_size;
    int T = model->seq_len;
    int C = model->config.channels;

    printf("\n[SEP] ln1 @ (layer=%d, b=%d, t=%d):\n", l, b, t);

    int printed = 0;
    while (printed < C)
    {
        printf("channels [%d]: ", printed);

        // find which device this b, t belongs to
        int device_id = -1;
        int channel_base = 0;
        for (int d = 0; d < 3; d++)
        {
            int C_dev = model->devices[d].allocated_channels;
            if (printed < channel_base + C_dev)
            {
                device_id = d;
                break;
            }
            channel_base += C_dev;
        }

        int local_c = printed - channel_base;
        gpt_device *dev = &model->devices[device_id];
        int C_dev = dev->allocated_channels;
        float *ln1 = dev->acts.ln1;

        float val = ln1[l * B * T * C_dev + (b * T + t) * C_dev + local_c];
        printf("%.4f\n", val);

        printed += 30;
    }
}
void layernorm_forward_sep_2(GPT2_Sep *model, int L, size_t B, size_t T)
{
    
    float mean_global[B * T]; 
    memset(mean_global, 0, sizeof(mean_global));

    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

        layernorm_sep_1(
            dev->acts.ln2_mean + (L * B * T),          
            dev->acts.residual2 + (L * B * T * C_dev), 
            B, T, C_dev);
    }

    
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int i = 0; i < 3; i++)
            {
                mean_global[b * T + t] += model->devices[i].acts.ln2_mean[L * B * T + b * T + t];
            }
            mean_global[b * T + t] /= model->config.channels;
        }
    }

   
    double var_global[B * T];
    memset(var_global, 0, sizeof(var_global));

    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

        layernorm_sep_2(
            dev->acts.ln2_rstd + (L * B * T),          
            dev->acts.residual2 + (L * B * T * C_dev), 
            mean_global, B, T, C_dev);
    }

    
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int i = 0; i < 3; i++)
            {
                var_global[b * T + t] += model->devices[i].acts.ln2_rstd[L * B * T + b * T + t];
            }
            var_global[b * T + t] /= model->config.channels;
            var_global[b * T + t] = 1.0f / sqrtf(var_global[b * T + t] + 1e-5f);
        }
    }

    
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

        layernorm_sep_3(
            dev->acts.ln2 + (L * B * T * C_dev),       
            dev->acts.residual2 + (L * B * T * C_dev), 
            mean_global, var_global,
            dev->params.ln2w + (L * C_dev), 
            dev->params.ln2b + (L * C_dev), 
            B, T, C_dev);
    }
}
void layernorm_forward_sep_2_modified(GPT2_Sep *model, int l, int B, int T)
{
    int C = model->config.channels;

    float mean_global[B * T];
    float rstd_global[B * T];

    
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float sum = 0.0f;

            for (int c = 0; c < C; c++) {
                int device_id = -1;
                int channel_base = 0;
                for (int d = 0; d < 3; d++) {
                    int C_dev = model->devices[d].allocated_channels;
                    if (c < channel_base + C_dev) {
                        device_id = d;
                        break;
                    }
                    channel_base += C_dev;
                }

                int local_c = c - channel_base;
                gpt_device *dev = &model->devices[device_id];
                float *x = dev->acts.residual2 + l * B * T * dev->allocated_channels;
                float val = x[(b * T + t) * dev->allocated_channels + local_c];
                sum += val;
            }

            float mean = sum / C;
            mean_global[b * T + t] = mean;

            float var = 0.0f;
            for (int c = 0; c < C; c++) {
                int device_id = -1;
                int channel_base = 0;
                for (int d = 0; d < 3; d++) {
                    int C_dev = model->devices[d].allocated_channels;
                    if (c < channel_base + C_dev) {
                        device_id = d;
                        break;
                    }
                    channel_base += C_dev;
                }

                int local_c = c - channel_base;
                gpt_device *dev = &model->devices[device_id];
                float *x = dev->acts.residual2 + l * B * T * dev->allocated_channels;
                float val = x[(b * T + t) * dev->allocated_channels + local_c];
                float diff = val - mean;
                var += diff * diff;
            }

            rstd_global[b * T + t] = 1.0f / sqrtf(var / C + 1e-5f);
        }
    }

    
    for (int d = 0; d < 3; d++) {
        gpt_device *dev = &model->devices[d];
        int C_dev = dev->allocated_channels;
        float *inp = dev->acts.residual2 + l * B * T * C_dev;
        float *out = dev->acts.ln2 + l * B * T * C_dev;
        float *weight = dev->params.ln2w + l * C_dev;
        float *bias = dev->params.ln2b + l * C_dev;

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float mean = mean_global[b * T + t];
                float rstd = rstd_global[b * T + t];

                float *x = inp + (b * T + t) * C_dev;
                float *y = out + (b * T + t) * C_dev;

                for (int i = 0; i < C_dev; i++) {
                    float norm = (x[i] - mean) * rstd;
                    y[i] = norm * weight[i] + bias[i];
                }
            }
        }
    }
}


void print_ln2_sep(GPT2_Sep *model, int l, int b, int t) {
    int C = model->config.channels;
    int T = model->seq_len;

    printf("\n[SEP] ln2 @ (layer=%d, b=%d, t=%d):\n", l, b, t);

    for (int c = 0; c < C; c += 30) {
        
        int device_id = -1;
        int channel_base = 0;
        for (int d = 0; d < 3; d++) {
            int C_dev = model->devices[d].allocated_channels;
            if (c < channel_base + C_dev) {
                device_id = d;
                break;
            }
            channel_base += C_dev;
        }

        if (device_id == -1) {
            printf("Error: cannot match channel %d to any device\n", c);
            continue;
        }

        int local_c = c - channel_base;
        gpt_device *dev = &model->devices[device_id];
        int C_dev = dev->allocated_channels;

        float *ln2 = dev->acts.ln2 + l * model->batch_size * T * C_dev;
        float val = ln2[(b * T + t) * C_dev + local_c];

        printf("channel [%d] = %.4f\n", c, val);
    }
}





void layernorm_forward_sep_3(GPT2_Sep *model, int L, size_t B, size_t T)
{
    
    float mean_global[B * T]; 
    memset(mean_global, 0, sizeof(mean_global));

    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

        layernorm_sep_1(
            dev->acts.lnf_mean,                              
            dev->acts.residual3 + ((L - 1) * B * T * C_dev), 
            B, T, C_dev);
    }

    
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int i = 0; i < 3; i++)
            {
                mean_global[b * T + t] += model->devices[i].acts.lnf_mean[b * T + t];
            }
            mean_global[b * T + t] /= model->config.channels;
        }
    }

    
    double var_global[B * T];
    memset(var_global, 0, sizeof(var_global));

    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

        layernorm_sep_2(
            dev->acts.lnf_rstd,                              
            dev->acts.residual3 + ((L - 1) * B * T * C_dev), 
            mean_global, B, T, C_dev);
    }

    
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int i = 0; i < 3; i++)
            {
                var_global[b * T + t] += model->devices[i].acts.lnf_rstd[b * T + t];
            }
            var_global[b * T + t] /= model->config.channels;
            var_global[b * T + t] = 1.0f / sqrtf(var_global[b * T + t] + 1e-5f);
        }
    }

    
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

        layernorm_sep_3(
            dev->acts.lnf,                                   
            dev->acts.residual3 + ((L - 1) * B * T * C_dev), 
            mean_global, var_global,
            dev->params.lnfw + (i * C_dev), 
            dev->params.lnfb + (i * C_dev), 
            B, T, C_dev);
    }
}
void layernorm_forward_sep_3_modified(GPT2_Sep *model, int L, int B, int T)
{
    int C = model->config.channels;

    float mean_global[B * T];
    float rstd_global[B * T];

    // Step 1: calculate global mean and inverse std
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float sum = 0.0f;

            for (int c = 0; c < C; c++) {
                int device_id = -1;
                int channel_base = 0;
                for (int d = 0; d < 3; d++) {
                    int C_dev = model->devices[d].allocated_channels;
                    if (c < channel_base + C_dev) {
                        device_id = d;
                        break;
                    }
                    channel_base += C_dev;
                }

                int local_c = c - channel_base;
                gpt_device *dev = &model->devices[device_id];
                float *x = dev->acts.residual3 + (L - 1) * B * T * dev->allocated_channels;
                float val = x[(b * T + t) * dev->allocated_channels + local_c];
                sum += val;
            }

            float mean = sum / C;
            mean_global[b * T + t] = mean;

            float var = 0.0f;
            for (int c = 0; c < C; c++) {
                int device_id = -1;
                int channel_base = 0;
                for (int d = 0; d < 3; d++) {
                    int C_dev = model->devices[d].allocated_channels;
                    if (c < channel_base + C_dev) {
                        device_id = d;
                        break;
                    }
                    channel_base += C_dev;
                }

                int local_c = c - channel_base;
                gpt_device *dev = &model->devices[device_id];
                float *x = dev->acts.residual3 + (L - 1) * B * T * dev->allocated_channels;
                float val = x[(b * T + t) * dev->allocated_channels + local_c];
                float diff = val - mean;
                var += diff * diff;
            }

            rstd_global[b * T + t] = 1.0f / sqrtf(var / C + 1e-5f);
        }
    }

    // Step 2: each device performs final normalization
    int offset=0;
    for (int d = 0; d < 3; d++) {
        gpt_device *dev = &model->devices[d];
        int C_dev = dev->allocated_channels;
        float *inp = dev->acts.residual3 + (L - 1) * B * T * C_dev;
        float *out = dev->acts.lnf;
        // float *weight = dev->params.lnfw + d * C_dev;
        // float *bias = dev->params.lnfb + d * C_dev;
        float *weight = dev->params.lnfw ;
        float *bias = dev->params.lnfb ;

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float mean = mean_global[b * T + t];
                float rstd = rstd_global[b * T + t];

                float *x = inp + (b * T + t) * C_dev;
                float *y = out + (b * T + t) * C_dev;

                for (int i = 0; i < C_dev; i++) {
                    float norm = (x[i] - mean) * rstd;
                    y[i] = norm * weight[i] + bias[i];
                }
            }
        }
    }
}

void print_lnf_sep(GPT2_Sep *model, int b, int t) {
    int C = model->config.channels;
    int T = model->seq_len;

    printf("\n[SEP] lnf (final LayerNorm output) @ (b=%d, t=%d):\n", b, t);

    for (int c = 0; c < C; c += 30) {
        
        int device_id = -1;
        int channel_base = 0;
        for (int d = 0; d < 3; d++) {
            int C_dev = model->devices[d].allocated_channels;
            if (c < channel_base + C_dev) {
                device_id = d;
                break;
            }
            channel_base += C_dev;
        }


        int local_c = c - channel_base;
        gpt_device *dev = &model->devices[device_id];
        int C_dev = dev->allocated_channels;

        float *lnf = dev->acts.lnf;
        float val = lnf[(b * T + t) * C_dev + local_c];

        printf("channel [%d] = %.4f \n", c, val);
    }
}

void matmul_forward_sep(float *global_out,
                        const float *inp, const float *weight, const float *bias,
                        int B, int T, int C_dev, int OC)
{

#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            int bt = b * T + t;
            float *global_out_pos = global_out + bt * OC; // 全局 QKV 存储位置, all qkv location

            for (int o = 0; o < OC; o++)
            {
                // float val = (bias != NULL) ? bias[o] : 0.0f;
                float val = 0.0f;
                for (int i = 0; i < C_dev; i++)
                {
                    // val += inp[bt * C_dev + i] * weight[o * C_dev + i];
                    global_out_pos[o] += inp[bt * C_dev + i] * weight[o * C_dev + i];
                }

            }
        }
    }
}

void matmul_forward_sep_modified(float *global_out,
                        const float *inp, const float *weight, int l,
                        int B, int T, int C_dev, int OC, int device_id)
{

//#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++)
    {
        for (int t = 0; t < T; t++)
        {
            int bt = b * T + t;
            float *global_out_pos = global_out + bt * OC; 

            for (int o = 0; o < OC; o++)
            {
                // float val = (bias != NULL) ? bias[o] : 0.0f;
                float val = 0.0f;
                for (int i = 0; i < C_dev; i++)
                {
                    // val += inp[bt * C_dev + i] * weight[o * C_dev + i];
                    global_out_pos[o] += inp[bt * C_dev + i] * weight[o * C_dev + i];
                    if (l==0&&b==0&&t == 0 && o == 0 && i>=2&&i<22 && device_id==2)
                    {
                        //printf("global_out_pos = %f, inp = %f, weight = %f\n", global_out_pos[o], inp[bt * C_dev + i], weight[o * C_dev + i]);
                    }
                }

                // #pragma omp atomic
                //                 global_out_pos[o] += val; 
            }
        }
    }
}

void matmul_forward_sep_qkv(GPT2_Sep *model, int l, size_t B, size_t T, size_t C, size_t OC)
{
    // Step 1: clear `global_qkv` buffer
    float *global_qkv = (float *)mallocCheck(B * T * OC * sizeof(float));
    memset(global_qkv, 0, B * T * OC * sizeof(float));

    // Step 2: each device computes its local QKV
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

        matmul_forward_sep_modified(
            global_qkv,                             // overall QKV 
            dev->acts.ln1 + (l * B * T * C_dev),    // LayerNorm output
            dev->params.qkvw + (l * 3 * C_dev * C), // QKV weight
            l,                                   // we do not need bias here, so pass l
            B, T, C_dev, OC,i);
    }

    // Step 3: add all `bias` to `global_qkv`
    int offset=0;
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;
        float *bias = dev->params.qkvb + (l * 3 * C_dev); 
       

#pragma omp parallel for collapse(2)
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float *global_qkv_pos = global_qkv + (b * T + t) * OC;
                float *bias_pos = bias;

                for (int j = 0; j < 3; j++) // `Q, K, V` 三个部分 
                {
                    for (int c = 0; c < C_dev; c++)
                    {
                        global_qkv_pos[j * C + c + offset] += bias_pos[j * C_dev + c];
                    }
                }
            }
        }
        offset+=C_dev;
    }

    if (l == 0)
    {
        int print_b = 0;
        int print_t = 0;

        float *global_qkv_pos = global_qkv + (print_b * T + print_t) * 3 * C;

        //Q
        // printf("\n[SEP] global_qkv @ (l=0, b=%d, t=%d), Q:\n", print_b, print_t);

        // for (int i = 0; i < C; i += 30) {
        //     printf("channel[%d] = %.4f\n", i, global_qkv_pos[i]);
        // }

        // // K
        // printf("\n[DEBUG] global_qkv @ (l=0, b=%d, t=%d), K:\n", print_b, print_t);
        // for (int i = 0; i < C; i += 30) {
        //     printf("K[%d] = %.4f\n", i, global_qkv_pos[C + i]);
        // }

        // // V
        // printf("\n[DEBUG] global_qkv @ (l=0, b=%d, t=%d), V:\n", print_b, print_t);
        // for (int i = 0; i < C; i += 30) {
        //     printf("V[%d] = %.4f\n", i, global_qkv_pos[2 * C + i]);
        // }
    }

    // Step 4: divide `global_qkv` into local QKV for each device
    offset=0;
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

#pragma omp parallel for
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float *global_qkv_pos = global_qkv + (b * T + t) * 3 * C; // 从全局 QKV 结果取值
                float *local_qkv = dev->acts.qkv + (l * B * T * 3 * C_dev) + (b * T + t) * 3 * C_dev;

                // copy qkv fractions to local_qkv
                memcpy(local_qkv, global_qkv_pos+offset, C_dev * sizeof(float));                     // Qs
                memcpy(local_qkv + C_dev, global_qkv_pos + C + offset, C_dev * sizeof(float));         // K
                memcpy(local_qkv + 2 * C_dev, global_qkv_pos + 2 * C+  offset, C_dev * sizeof(float)); // V
            }
        }
        offset+=C_dev;
    }

    free(global_qkv); // relaese the global QKV buffer
}

void print_qkv_sep(GPT2_Sep *model, int l, int b, int t, const char *tag)
{
    int C = model->config.channels;

    printf("\n[SEP] qkv @ (layer=%d, b=%d, t=%d), tag=%s:\n", l, b, t, tag);
    size_t B = model->batch_size;
    size_t T = model->seq_len;
    // tag: "Q", "K", or "V"
    int offset_multiplier = 0;
    if (strcmp(tag, "K") == 0)
        offset_multiplier = 1;
    else if (strcmp(tag, "V") == 0)
        offset_multiplier = 2;

    int printed = 0;
    while (printed < C)
    {
        printf("channels [%d]: ", printed);

        // find which device this b, t belongs to
        int device_id = -1;
        int channel_base = 0;
        for (int d = 0; d < 3; d++)
        {
            int C_dev = model->devices[d].allocated_channels;
            if (printed < channel_base + C_dev)
            {
                device_id = d;
                break;
            }
            channel_base += C_dev;
        }

        int local_c = printed - channel_base;
        gpt_device *dev = &model->devices[device_id];
        int C_dev = dev->allocated_channels;

        float *qkv = dev->acts.qkv;

        // index into the qkv array
        float val = qkv[l * B * T * 3 * C_dev + (b * T + t) * 3 * C_dev + offset_multiplier * C_dev + local_c];

        printf("%.4f\n", val);
        printed += 30;
    }
}

void attention_forward_sep(GPT2_Sep *model, int l, size_t B, size_t T, size_t C, size_t NH)
{
    int C3 = 3 * C;                
    int hs = C / NH;               
    float scale = 1.0 / sqrtf(hs); // scaling factor for attention scores

    // Step 1: calculate `preatt` and `att` for each device
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;
        int local_heads = C_dev / hs; 

#pragma omp parallel for collapse(3)
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                for (int h = 0; h < local_heads; h++)
                {                                                                                       
                    int head_offset = h * hs;                                                           
                    float *query_t = dev->acts.qkv + (l * B * T * 3*C_dev) + (b * T + t) * 3*C_dev + head_offset; // Q
                    float *preatt_bth = dev->acts.preatt + (l * B * local_heads * T * T) + (b * local_heads * T * T) + (h * T * T) + (t * T);
                    float *att_bth = dev->acts.att + (l * B * local_heads * T * T) + (b * local_heads * T * T) + (h * T * T) + (t * T);

                    float maxval = -10000.0f;
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        float val = 0.0f;
                        float *key_t2 = dev->acts.qkv + (l * B * T * 3*C_dev) + (b * T + t2) * 3*C_dev + head_offset + C_dev; // K

                        //  Q · K^T
                        for (int k = 0; k < hs; k++)
                        {
                            val += query_t[k] * key_t2[k];
                        }

                        val *= scale;
                        if (val > maxval)
                        {
                            maxval = val;
                        }
                        preatt_bth[t2] = val;
                    }

                    // Softmax 
                    float expsum = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        float expv = expf(preatt_bth[t2] - maxval);
                        expsum += expv;
                        att_bth[t2] = expv;
                        
                    }

                    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                    for (int t2 = 0; t2 < T; t2++)
                    {
                        if (t2 <= t)
                        {
                            att_bth[t2] *= expsum_inv;
                        }
                        else
                        {
                            // causal attention mask. not strictly necessary to set to zero here
                            // only doing this explicitly for debugging and checking to PyTorch
                            att_bth[t2] = 0.0f;
                        }
                    }

                    float *out_bth = dev->acts.atty + (l * B * T * C_dev) + (b * T + t) * C_dev + head_offset;

                    // initialize `out_bth`
                    for (int i = 0; i < hs; i++)
                    {
                        out_bth[i] = 0.0f;
                    } // set to zero

                    for (int t2 = 0; t2 <= t; t2++)
                    {
                        float *value_t2 = dev->acts.qkv + (l * B * T * 3*C_dev) + (b * T + t2) * 3*C_dev + head_offset + 2 * C_dev; // V
                        float att_btht2 = att_bth[t2];

                        for (int i = 0; i < hs; i++)
                        {
                            out_bth[i] += att_btht2 * value_t2[i];
                        }
                    }
                }
            }
        }
    }
}
void print_atty_sep(GPT2_Sep *model, int l, int b, int t) {
    int C = model->config.channels;
    int T = model->seq_len;

    printf("\n[SEP] atty @ (layer=%d, b=%d, t=%d):\n", l, b, t);

    for (int c = 0; c < C; c += 30) {
        // find which device this channel belongs to
        int device_id = -1;
        int channel_base = 0;
        for (int d = 0; d < 3; d++) {
            int C_dev = model->devices[d].allocated_channels;
            if (c < channel_base + C_dev) {
                device_id = d;
                break;
            }
            channel_base += C_dev;
        }

        if (device_id == -1) {
            printf("channel %d cannot match any device!\n", c);
            continue;
        }

        int local_c = c - channel_base;
        gpt_device *dev = &model->devices[device_id];
        int C_dev = dev->allocated_channels;

        float *atty = dev->acts.atty + l * model->batch_size * T * C_dev;
        float val = atty[(b * T + t) * C_dev + local_c];
        printf("channel [%d] = %.4f\n", c, val);
    }
}

void matmul_forward_sep_attproj(GPT2_Sep *model, int l, size_t B, size_t T, size_t C, size_t OC)
{
    // Step 1: Initialize global Attention Projection result
    float *global_attproj = (float *)mallocCheck(B * T * OC * sizeof(float));
    memset(global_attproj, 0, B * T * OC * sizeof(float));

    // Step 2: Each device computes its partial_attproj and accumulates to global_attproj
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

        matmul_forward_sep(
            global_attproj,                         // global Attention Projection result
            dev->acts.atty + (l * B * T * C_dev),   // output of Self-Attention on this device
            dev->params.attprojw + (l * C_dev * C), // Attention Projection weights
            NULL,                                   // no bias applied here
            B, T, C_dev, OC);
    }

    // Step 3: Add bias from each device into global_attproj
    int offset = 0;
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;
        float *bias = dev->params.attprojb + (l * C_dev); // device-specific bias

#pragma omp parallel for collapse(2)
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float *global_attproj_pos = global_attproj + (b * T + t) * OC;
                float *bias_pos = bias;

                for (int c = 0; c < C_dev; c++) // add only the partial bias handled by this device
                {
                    global_attproj_pos[c + offset] += bias_pos[c];
                }
            }
        }
        offset += C_dev;
    }

    // Step 4: Split global_attproj into per-device segments by C_dev
    offset = 0;
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

#pragma omp parallel for
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float *global_attproj_pos = global_attproj + (b * T + t) * C; // source from global attproj
                float *local_attproj = dev->acts.attproj + (l * B * T * C_dev) + (b * T + t) * C_dev;

                // copy C_dev-dimension segment to local device
                memcpy(local_attproj, global_attproj_pos + offset, C_dev * sizeof(float));
            }
        }
        offset += C_dev;
    }

    free(global_attproj); // free global buffer
}


void print_attproj_sep(GPT2_Sep *model, int l, int b, int t) {
    int C = model->config.channels;
    int T = model->seq_len;

    printf("\n[SEP] attproj @ (layer=%d, b=%d, t=%d):\n", l, b, t);

    for (int c = 0; c < C; c += 30) {
        
        int device_id = -1;
        int channel_base = 0;
        for (int d = 0; d < 3; d++) {
            int C_dev = model->devices[d].allocated_channels;
            if (c < channel_base + C_dev) {
                device_id = d;
                break;
            }
            channel_base += C_dev;
        }

        if (device_id == -1) {
            printf("Error: channel %d cannot match any device\n", c);
            continue;
        }

        int local_c = c - channel_base;
        gpt_device *dev = &model->devices[device_id];
        int C_dev = dev->allocated_channels;

        float *attproj = dev->acts.attproj + l * model->batch_size * T * C_dev;
        float val = attproj[(b * T + t) * C_dev + local_c];

        printf("channel [%d] = %.4f\n", c, val);
    }
}


void matmul_forward_sep_fc1(GPT2_Sep *model, int l, size_t B, size_t T, size_t C, size_t OC)
{
    // Step 1: Initialize global FC1 output buffer (`fch`)
    float *global_fch = (float *)mallocCheck(B * T * OC * sizeof(float));
    memset(global_fch, 0, B * T * OC * sizeof(float));

    // Step 2: Each device computes its partial_fch and accumulates into global_fch
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

        matmul_forward_sep(
            global_fch,                            // global fch result buffer
            dev->acts.ln2 + (l * B * T * C_dev),   // LayerNorm output (ln2)
            dev->params.fcw + (l * 4 * C_dev * C), // FC1 weights (4*C by C)
            NULL,                                  // no bias added here
            B, T, C_dev, OC);
    }

    // Step 3: Add bias from each device into global_fch
    int offset = 0;
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;
        float *bias = dev->params.fcb + (l * 4 * C_dev); // bias for this device

#pragma omp parallel for collapse(2)
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float *global_fch_pos = global_fch + (b * T + t) * OC;
                float *bias_pos = bias;

                for (int j = 0; j < 4; j++) // loop over 4*C dimensions
                {
                    for (int c = 0; c < C_dev; c++)
                    {
                        global_fch_pos[j * C + c + offset] += bias_pos[j * C_dev + c];
                    }
                }
            }
        }
        offset += C_dev;
    }

    // Step 4: Split global_fch into per-device buffers `fch`
    offset = 0;
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

#pragma omp parallel for
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float *global_fch_pos = global_fch + (b * T + t) * 4 * C; // source from global buffer
                float *local_fch = dev->acts.fch + (l * B * T * 4 * C_dev) + (b * T + t) * 4 * C_dev;

                // Copy the 4 segments (C_dev each) to local device buffer
                memcpy(local_fch, global_fch_pos + offset, C_dev * sizeof(float));                     // first segment
                memcpy(local_fch + C_dev, global_fch_pos + offset + C, C_dev * sizeof(float));         // second segment
                memcpy(local_fch + 2 * C_dev, global_fch_pos + offset + 2 * C, C_dev * sizeof(float)); // third segment
                memcpy(local_fch + 3 * C_dev, global_fch_pos + offset + 3 * C, C_dev * sizeof(float)); // fourth segment
            }
        }
        offset += C_dev;
    }

    free(global_fch); // free global buffer
}


void print_fch_sep(GPT2_Sep *model, int l, int b, int t) {
    int C = model->config.channels;
    int T = model->seq_len;

    printf("\n[SEP] fch (FC1 output) @ (layer=%d, b=%d, t=%d):\n", l, b, t);

    //打印alloc_channels
    for (int d = 0; d < 3; d++) {
       // printf("device[%d] alloc_channels = %d\n", d, model->devices[d].allocated_channels);
    }

    for (int c = 0; c < 4 * C; c += 100) {

        
        int device_id = -1;
        int channel_base = 0;
        int block = c / C;           
        int c_in_block = c- C*block;      // channel number in the block: 0,1,2,...,C-1

        for (int d = 0; d < 3; d++) {
            int C_dev = model->devices[d].allocated_channels;
            if (c_in_block < channel_base +  C_dev) {
                device_id = d;
                break;
            }
            channel_base += C_dev;
        }

        //printf("device_id = %d, block = %d, c_in_block = %d\n", device_id, block, c_in_block);

        
        gpt_device *dev = &model->devices[device_id];
        int C_dev = dev->allocated_channels;

        float *fch = dev->acts.fch + l * model->batch_size * T * 4 * C_dev;
        float val = fch[(b * T + t) * 4 * C_dev +block*C_dev+ c_in_block-channel_base];

        printf("channel [%d] = %.4f\n", c, val);
    }
}



void matmul_forward_sep_fc2(GPT2_Sep *model, int l, size_t B, size_t T, size_t C, size_t OC)
{
    
    float *global_fcproj = (float *)mallocCheck(B * T * OC * sizeof(float));
    memset(global_fcproj, 0, B * T * OC * sizeof(float));

    
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

        matmul_forward_sep(
            global_fcproj,                                // global fcproj result buffer
            dev->acts.fch_gelu + (l * B * T * 4 * C_dev), // GELU-activated output of FC1
            dev->params.fcprojw + (l * C_dev * 4 * C),    // FC2 weight matrix
            NULL,                                         // no bias applied here
            B, T, 4 * C_dev, OC);
    }

    
    int offset = 0;
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;
        float *bias = dev->params.fcprojb + (l * C_dev); 

#pragma omp parallel for collapse(2)
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float *global_fcproj_pos = global_fcproj + (b * T + t) * OC;
                float *bias_pos = bias;

                for (int c = 0; c < C_dev; c++) 
                {
                    global_fcproj_pos[c+offset] += bias_pos[c];
                }
            }
        }
        offset += C_dev;
    }

    offset = 0;

    
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

#pragma omp parallel for
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float *global_fcproj_pos = global_fcproj + (b * T + t) * C; 
                float *local_fcproj = dev->acts.fcproj + (l * B * T * C_dev) + (b * T + t) * C_dev;

                
                memcpy(local_fcproj, global_fcproj_pos+offset, C_dev * sizeof(float));
            }
        }
        offset += C_dev;
    }

    free(global_fcproj); 
}

void print_fcproj_sep(GPT2_Sep *model, int l, int b, int t) {
    int C = model->config.channels;
    int T = model->seq_len;

    printf("\n[SEP] fcproj (FC2 output) @ (layer=%d, b=%d, t=%d):\n", l, b, t);

    for (int c = 0; c < C; c += 30) {
        
        int device_id = -1;
        int channel_base = 0;
        for (int d = 0; d < 3; d++) {
            int C_dev = model->devices[d].allocated_channels;
            if (c < channel_base + C_dev) {
                device_id = d;
                break;
            }
            channel_base += C_dev;
        }

        if (device_id == -1) {
            printf("channel %d cannot match any device!\n", c);
            continue;
        }

        int local_c = c - channel_base;
        gpt_device *dev = &model->devices[device_id];
        int C_dev = dev->allocated_channels;

        float *fcproj = dev->acts.fcproj + l * model->batch_size * T * C_dev;
        float val = fcproj[(b * T + t) * C_dev + local_c];

        printf("channel [%d] = %.4f\n", c, val);
    }
}

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI) // M_PI represents the value of pi (π) in the math library

void gelu_forward_sep(GPT2_Sep *model, int l, size_t B, size_t T)
{
    
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels; 
        int C4_dev = 4 * C_dev;              

#pragma omp parallel for
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float *inp_bt = dev->acts.fch + (l * B * T * C4_dev) + (b * T + t) * C4_dev;      
                float *out_bt = dev->acts.fch_gelu + (l * B * T * C4_dev) + (b * T + t) * C4_dev; 

                // GELU calculation: out[i] = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x^3) ) )
                for (int c = 0; c < C4_dev; c++)
                {
                    float x = inp_bt[c];
                    float cube = 0.044715f * x * x * x;
                    out_bt[c] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
                }
            }
        }
    }
}

void  print_fch_gelu_sep(GPT2_Sep *model, int l, int b, int t) {
    int C = model->config.channels;
    int T = model->seq_len;

    printf("\n[SEP] fch_gelu (GELU output) @ (layer=%d, b=%d, t=%d):\n", l, b, t);

    for (int c = 0; c < 4 * C; c += 100) {
        int block = c / C;              
        int c_in_block = c % C;         

        int device_id = -1;
        int channel_base = 0;

        for (int d = 0; d < 3; d++) {
            int C_dev = model->devices[d].allocated_channels;
            if (c_in_block < channel_base + C_dev) {
                device_id = d;
                break;
            }
            channel_base += C_dev;
        }

        int local_c = c_in_block - channel_base;
        gpt_device *dev = &model->devices[device_id];
        int C_dev = dev->allocated_channels;

        float *fch_gelu = dev->acts.fch_gelu + l * model->batch_size * T * 4 * C_dev;
        float val = fch_gelu[(b * T + t) * 4 * C_dev + block * C_dev + local_c];

        printf("channel [%d] = %.4f \n", c, val);
    }
}


void matmul_forward_sep_logits(GPT2_Sep *model, size_t B, size_t T, size_t C, size_t Vp)
{
    
    float *global_logits = (float *)mallocCheck(B * T * Vp * sizeof(float));
    memset(global_logits, 0, B * T * Vp * sizeof(float));

   
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

        matmul_forward_sep(
            global_logits,   
            dev->acts.lnf,   
            dev->params.wte, 
            NULL,            
            B, T, C_dev, Vp);
    }

    
    size_t offset = 0; 
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;
        size_t Vp_dev = (size_t)((float)Vp * ((float)C_dev / C)); 

#pragma omp parallel for
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float *global_logits_pos = global_logits + (b * T + t) * Vp + offset;
                float *local_logits = dev->acts.logits + (b * T + t) * Vp_dev;

                memcpy(local_logits, global_logits_pos, Vp_dev * sizeof(float));
            }
        }

        offset += Vp_dev; 
    }

    free(global_logits); 
}

void print_logits_sep(GPT2_Sep *model, int b, int t) {
    int Vp = model->config.padded_vocab_size;
    int T = model->seq_len;
    size_t C = model->config.channels;

    printf("\n[SEP] logits @ (b=%d, t=%d):\n", b, t);

    for (int v = 0; v < Vp; v += 1300) {
        
        int device_id = -1;
        int vocab_base = 0;
        for (int d = 0; d < 3; d++) {
            int C_dev = model->devices[d].allocated_channels;
            size_t Vp_dev = (size_t)((float)Vp * ((float)C_dev / C)); 
            if (v < vocab_base + Vp_dev) {
                device_id = d;
                break;
            }
            vocab_base += Vp_dev;
        }


        int local_v = v - vocab_base;
        gpt_device *dev = &model->devices[device_id];
        int C_dev = dev->allocated_channels;
        size_t Vp_dev = (size_t)((float)Vp * ((float)C_dev / C)); 
        
        float *logits = dev->acts.logits;
        float val = logits[(b * T + t) * Vp_dev + local_v];

        printf("vocab[%d] = %.4f\n", v, val);
    }
}

void residual_forward_sep_attproj(GPT2_Sep *model, int L, size_t B, size_t T, size_t C)
{
    
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels; 

#pragma omp parallel for
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                int idx = (b * T + t) * C_dev;
                float *out = dev->acts.residual2 + (L * B * T * C_dev) + idx; // `residual2[L]`
                float *inp1;

                // `L=0` ，`residual` equals `encoded`
                if (L == 0)
                    inp1 = dev->acts.encoded + idx;
                else
                    inp1 = dev->acts.residual3 + ((L - 1) * B * T * C_dev) + idx; //take `residual[L-1]`** previous Lthlayer residual3

                float *inp2 = dev->acts.attproj + (L * B * T * C_dev) + idx; // `attproj[L]`

                for (int c = 0; c < C_dev; c++)
                {
                    out[c] = inp1[c] + inp2[c];
                }
            }
        }
    }
}

void print_residual2_sep(GPT2_Sep *model, int l, int b, int t) {
    int C = model->config.channels;
    int T = model->seq_len;

    printf("\n[SEP] residual2 @ (layer=%d, b=%d, t=%d):\n", l, b, t);

    for (int c = 0; c < C; c += 30) {
        
        int device_id = -1;
        int channel_base = 0;
        for (int d = 0; d < 3; d++) {
            int C_dev = model->devices[d].allocated_channels;
            if (c < channel_base + C_dev) {
                device_id = d;
                break;
            }
            channel_base += C_dev;
        }

        if (device_id == -1) {
            printf("Error: channel %d cannot match any device!\n", c);
            continue;
        }

        int local_c = c - channel_base;
        gpt_device *dev = &model->devices[device_id];
        int C_dev = dev->allocated_channels;

        float *residual2 = dev->acts.residual2 + l * model->batch_size * T * C_dev;
        float val = residual2[(b * T + t) * C_dev + local_c];

        printf("channel [%d] = %.4f\n", c, val);
    }
}


void residual_forward_sep_fcproj(GPT2_Sep *model, int L, size_t B, size_t T, size_t C)
{
    
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels; 

#pragma omp parallel for
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                int idx = (b * T + t) * C_dev;
                float *out = dev->acts.residual3 + (L * B * T * C_dev) + idx;  // `residual3[L]`
                float *inp1 = dev->acts.residual2 + (L * B * T * C_dev) + idx; // `residual2[L]`
                float *inp2 = dev->acts.fcproj + (L * B * T * C_dev) + idx;    // `fcproj[L]`

                for (int c = 0; c < C_dev; c++)
                {
                    out[c] = inp1[c] + inp2[c];
                }
            }
        }
    }
}
void print_residual3_sep(GPT2_Sep *model, int l, int b, int t) {
    int C = model->config.channels;
    int T = model->seq_len;

    printf("\n[SEP] residual3 @ (layer=%d, b=%d, t=%d):\n", l, b, t);

    for (int c = 0; c < C; c += 30) {
       
        int device_id = -1;
        int channel_base = 0;
        for (int d = 0; d < 3; d++) {
            int C_dev = model->devices[d].allocated_channels;
            if (c < channel_base + C_dev) {
                device_id = d;
                break;
            }
            channel_base += C_dev;
        }

        if (device_id == -1) {
            printf("Error: channel %d cannot match any device!\n", c);
            continue;
        }

        int local_c = c - channel_base;
        gpt_device *dev = &model->devices[device_id];
        int C_dev = dev->allocated_channels;

        float *residual3 = dev->acts.residual3 + l * model->batch_size * T * C_dev;
        float val = residual3[(b * T + t) * C_dev + local_c];

        printf("channel [%d] = %.4f \n", c, val);
    }
}



// Gaussian Error Linear Unit

// We want to use -Ofast optimization globally, but GeLU is numerically unstable under it,
// so we disable aggressive optimizations locally here (#168)
#pragma float_control(precise, on, push) 
// In 'precise' mode, the compiler disables aggressive floating-point optimizations
// to strictly comply with IEEE 754 standards.

#if defined(__GNUC__) && !defined(__clang__)

__attribute__((optimize("no-finite-math-only")))
// Disable GCC's "finite math only" optimization
// GCC may assume all floating-point operations produce finite results (no Inf or NaN),
// which allows skipping some NaN/Inf checks to improve performance.
// However, this may be unsafe in ML or scientific code where NaN or Inf may arise.
#endif


void softmax_forward_sep(GPT2_Sep *model, size_t B, size_t T, size_t V, size_t Vp)
{

    float *global_maxval = (float *)mallocCheck(B * T * sizeof(float));
    float *global_sum = (float *)mallocCheck(B * T * sizeof(float));
   // memset(global_maxval, -10000.0f, B * T * sizeof(float)); 
    for (int i = 0; i < B * T; i++) {
        global_maxval[i] = -10000.0f;
    }
    memset(global_sum, 0, B * T * sizeof(float));            

    
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels; 
        size_t V_Dev = (size_t)((float)Vp * ((float)C_dev / model->config.channels));

        printf("V_Dev:%ld from device%d \n", V_Dev,i);
        

        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float *logits_bt = dev->acts.logits + (b * T + t) * V_Dev; 
                if(i==2){
                    V_Dev -= Vp - V;
                }
                
                float local_maxval = -10000.0f;

                
                for (int j = 0; j < V_Dev; j++)
                {   

                    if(b==3&&t==8&&j%1000==0)
                    {
                        //printf("logits_bt[j=%d] from device[%d]:%f\n",j, i, logits_bt[j]);
                    }
                    //printf("logits_bt[j=%d] from device[%d]:%f\n",j, i, logits_bt[j]);
                    if (logits_bt[j] > local_maxval)
                    {
                        local_maxval = logits_bt[j];

                        //printf("new local_maxval:%f\n", local_maxval);
                    }
                    if(logits_bt[j] ==0)
                    {
                        //printf("logits_bt is 0 at b=%d, t=%d ,j=%d in device %d \n", b, t,j,i);
                    }
                }

                    if(local_maxval ==0)
                    {
                       // printf("local_maxval is 0 at b=%d, t=%d\n", b, t);
                    }
                    
                    if (local_maxval > global_maxval[b * T + t])
                    {
                        global_maxval[b * T + t] = local_maxval;
                        
                    }
                
            }
        }
        printf("global_maxval in 2244 line [200]:%f\n", global_maxval[200]);
    }

    
    for (int i = 0; i < 3; i++)
    {
        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;
        size_t V_Dev = (size_t)((float)Vp * ((float)C_dev / model->config.channels));


        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float *logits_bt = dev->acts.logits + (b * T + t) * V_Dev;
                float local_sum = 0.0f;
                float maxval = global_maxval[b * T + t]; 
                if(i==2){
                    V_Dev -= Vp - V;
                }

                for (int j = 0; j < V_Dev; j++)
                {
                    local_sum += expf(logits_bt[j] - maxval);
                }

                global_sum[b * T + t] += local_sum;
            }
        }
    }
    printf("global_sum[200]:%f\n", global_sum[200]);
    printf("global_maxval[200]:%f\n", global_maxval[200]);


    float device_prob_check[B][T][3];                        
    memset(device_prob_check, 0, sizeof(device_prob_check)); 

    for (int i = 0; i < 3; i++)
    {

        gpt_device *dev = &model->devices[i];
        int C_dev = dev->allocated_channels;

        size_t V_Dev = (size_t)((float)Vp * ((float)C_dev / model->config.channels));

        // #pragma omp parallel for collapse(2)
        for (int b = 0; b < B; b++)
        {
            for (int t = 0; t < T; t++)
            {
                float *logits_bt = dev->acts.logits + (b * T + t) * V_Dev;
                float *probs_bt = dev->acts.probs + (b * T + t) * V_Dev;
                float maxval = global_maxval[b * T + t];
                float sum = global_sum[b * T + t]; 
                float prob_check = 0.0f;

                if(i==2){
                    V_Dev -= Vp - V;
                }

                for (int j = 0; j < V_Dev; j++)
                {
                    // printf("logits_bt[j]:%f\t",logits_bt[j]);
                    probs_bt[j] = expf(logits_bt[j] - maxval) / sum;
                    
                    // global_prob_check[b * T + t] += probs_bt[j];
                }
                if(i==2){
                    for(int j = V_Dev; j < Vp; j++){
                        probs_bt[j] = 0.0f;
                    }
                }


                device_prob_check[b][t][i] = prob_check;

            }
        }
    }


}

void softmax_forward_sep_modified(GPT2_Sep *model, size_t B, size_t T, size_t V, size_t Vp) {
    int C = model->config.channels;

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            
            float maxval = -1e30f;
            for (int i = 0; i < V; i++) {
                int device_id = -1;
                int vocab_base = 0;
                for (int d = 0; d < 3; d++) {
                    int C_dev = model->devices[d].allocated_channels;
                    size_t Vp_dev = (size_t)((float)Vp * ((float)C_dev / C));
                    if (i < vocab_base + Vp_dev) {
                        device_id = d;
                        int local_i = i - vocab_base;
                        float *logits = model->devices[d].acts.logits;
                        float val = logits[(b * T + t) * Vp_dev + local_i];
                        if (val > maxval) maxval = val;
                        break;
                    }
                    vocab_base += Vp_dev;
                }
            }

            // sum = ∑exp(x - max)
            float sum = 0.0f;
            for (int i = 0; i < V; i++) {
                int device_id = -1;
                int vocab_base = 0;
                for (int d = 0; d < 3; d++) {
                    int C_dev = model->devices[d].allocated_channels;
                    size_t Vp_dev = (size_t)((float)Vp * ((float)C_dev / C));
                    if (i < vocab_base + Vp_dev) {
                        device_id = d;
                        int local_i = i - vocab_base;
                        float *logits = model->devices[d].acts.logits;
                        float val = logits[(b * T + t) * Vp_dev + local_i];
                        sum += expf(val - maxval);
                        break;
                    }
                    vocab_base += Vp_dev;
                }
            }

            // write back probs[i] = exp(x - max) / sum
            for (int i = 0; i < Vp; i++) {
                int device_id = -1;
                int vocab_base = 0;
                for (int d = 0; d < 3; d++) {
                    int C_dev = model->devices[d].allocated_channels;
                    size_t Vp_dev = (size_t)((float)Vp * ((float)C_dev / C));
                    if (i < vocab_base + Vp_dev) {
                        device_id = d;
                        int local_i = i - vocab_base;
                        float *logits = model->devices[d].acts.logits;
                        float *probs = model->devices[d].acts.probs;

                        float val = logits[(b * T + t) * Vp_dev + local_i];
                        if (i < V) {
                            probs[(b * T + t) * Vp_dev + local_i] = expf(val - maxval) / sum;
                        } else {
                            probs[(b * T + t) * Vp_dev + local_i] = 0.0f;
                        }
                        break;
                    }
                    vocab_base += Vp_dev;
                }
            }
        }
    }
}

void print_probs_sep(GPT2_Sep *model, int b, int t) {
    int Vp = model->config.padded_vocab_size;
    int T = model->seq_len;
    int C = model->config.channels;

    printf("\n[SEP] probs (softmax output) @ (b=%d, t=%d):\n", b, t);

    for (int v = 0; v < Vp; v += 1300) {
        
        int device_id = -1;
        int vocab_base = 0;

        for (int d = 0; d < 3; d++) {
            int C_dev = model->devices[d].allocated_channels;
            size_t Vp_dev = (size_t)((float)Vp * ((float)C_dev / C));
            if (v < vocab_base + Vp_dev) {
                device_id = d;
                break;
            }
            vocab_base += Vp_dev;
        }

        int local_v = v - vocab_base;
        gpt_device *dev = &model->devices[device_id];
        int C_dev = dev->allocated_channels;
        size_t Vp_dev = (size_t)((float)Vp * ((float)C_dev / C));

        float *probs = dev->acts.probs;
        float val = probs[(b * T + t) * Vp_dev + local_v];

        printf("vocab[%d] = %.6f \n", v, val);
    }
}

// ----------------------------------------------------------------------------
// GPT-2 model definition

void fill_in_activation_sizes(size_t *act_sizes, GPT2Config config, int B, int T)
{
    size_t C = config.channels;
    size_t NH = config.num_heads;
    size_t L = config.num_layers;
    size_t Vp = config.padded_vocab_size;
    act_sizes[0] = B * T * C;          // encoded
    act_sizes[1] = L * B * T * C;      // ln1
    act_sizes[2] = L * B * T;          // ln1_mean
    act_sizes[3] = L * B * T;          // ln1_rstd
    act_sizes[4] = L * B * T * 3 * C;  // qkv
    act_sizes[5] = L * B * T * C;      // atty
    act_sizes[6] = L * B * NH * T * T; // preatt
    act_sizes[7] = L * B * NH * T * T; // att
    act_sizes[8] = L * B * T * C;      // attproj
    act_sizes[9] = L * B * T * C;      // residual2
    act_sizes[10] = L * B * T * C;     // ln2
    act_sizes[11] = L * B * T;         // ln2_mean
    act_sizes[12] = L * B * T;         // ln2_rstd
    act_sizes[13] = L * B * T * 4 * C; // fch
    act_sizes[14] = L * B * T * 4 * C; // fch_gelu
    act_sizes[15] = L * B * T * C;     // fcproj
    act_sizes[16] = L * B * T * C;     // residual3
    act_sizes[17] = B * T * C;         // lnf
    act_sizes[18] = B * T;             // lnf_mean
    act_sizes[19] = B * T;             // lnf_rstd
    // 
    act_sizes[20] = B * T * Vp; // logits
    act_sizes[21] = B * T * Vp; // probs
    act_sizes[22] = B * T;      // losses
}

float *malloc_and_point_activations(ActivationTensors *acts, size_t *act_sizes)
{
    size_t num_activations = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++)
    {
        num_activations += act_sizes[i];
    }
    float *acts_memory = (float *)mallocCheck(num_activations * sizeof(float));
    float **ptrs[] = {
        &acts->encoded, &acts->ln1, &acts->ln1_mean, &acts->ln1_rstd, &acts->qkv, &acts->atty,
        &acts->preatt, &acts->att, &acts->attproj, &acts->residual2, &acts->ln2, &acts->ln2_mean,
        &acts->ln2_rstd, &acts->fch, &acts->fch_gelu, &acts->fcproj, &acts->residual3, &acts->lnf,
        &acts->lnf_mean, &acts->lnf_rstd, &acts->logits, &acts->probs, &acts->losses};
    float *acts_memory_iterator = acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++)
    {
        *(ptrs[i]) = acts_memory_iterator;
        acts_memory_iterator += act_sizes[i];
    }
    return acts_memory;
}

void gpt2_forward_sep(GPT2_Sep *model, int *inputs, size_t B, size_t T)
{

    // insure model is initialized
    if (model->all_params_memory == NULL)
    {
        printf("Error: Model was not initialized properly.\n");
        exit(1);
    }

    // read model configuration
    size_t V = model->config.vocab_size;
    size_t Vp = model->config.padded_vocab_size;
    size_t L = model->config.num_layers;
    size_t NH = model->config.num_heads;
    size_t C = model->config.channels;

    // check inputs
    for (int i = 0; i < B * T; i++)
    {
        if (0 > inputs[i] || inputs[i] > V)
        {
            printf("Error: input index out of range: %d\n", inputs[i]);
            exit(1);
        }
        assert(0 <= inputs[i] && inputs[i] < V);
    }

    // allocate activations memory if not already allocated
    if (model->all_acts_memory == NULL)
    {
        fill_in_activation_sizes(model->act_sizes, model->config, B, T);
        model->all_acts_memory = malloc_and_point_activations(&model->all_acts, model->act_sizes);
        // record B T
        model->batch_size = B;
        model->seq_len = T;

        

        size_t num_activations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++)
        {
            num_activations += model->act_sizes[i];
        }
        // printf("Total model number of activations: %zu\n", num_activations);
        model->num_activations = num_activations;
        // printf("num_activations: %zu\n", num_activations);

        
        for (int i = 0; i < 3; i++)
        {
            // printf("Device %d: Allocating activations for B=%zu, T=%zu, C=%d\n", i, B, T, model->devices[i].allocated_channels);

            
            size_t device_act_size = 0;
            for (size_t j = 0; j < NUM_ACTIVATION_TENSORS; j++)
            {
                // if (j == 20 || j == 21 || j == 22)
                // { 
                //     device_act_size += model->act_sizes[j];
                // }
                // else
                // {
                device_act_size += (size_t)(model->act_sizes[j] * ((float)model->devices[i].allocated_channels / model->config.channels));
                // }
            }

            
            model->devices[i].device_acts_memory = (float *)mallocCheck(device_act_size * sizeof(float));

            
            float **act_ptrs[] = {
                &model->devices[i].acts.encoded, &model->devices[i].acts.ln1, &model->devices[i].acts.ln1_mean, &model->devices[i].acts.ln1_rstd,
                &model->devices[i].acts.qkv, &model->devices[i].acts.atty, &model->devices[i].acts.preatt, &model->devices[i].acts.att,
                &model->devices[i].acts.attproj, &model->devices[i].acts.residual2, &model->devices[i].acts.ln2, &model->devices[i].acts.ln2_mean,
                &model->devices[i].acts.ln2_rstd, &model->devices[i].acts.fch, &model->devices[i].acts.fch_gelu, &model->devices[i].acts.fcproj,
                &model->devices[i].acts.residual3, &model->devices[i].acts.lnf, &model->devices[i].acts.lnf_mean, &model->devices[i].acts.lnf_rstd,
                &model->devices[i].acts.logits, &model->devices[i].acts.probs, &model->devices[i].acts.losses};

            float *device_act_iterator = model->devices[i].device_acts_memory;

            size_t offset = 0;

            for (size_t j = 0; j < NUM_ACTIVATION_TENSORS; j++)
            {
                size_t tensor_size = model->act_sizes[j];
                size_t device_tensor_size = 0;

                // if (j == 20 || j == 21 || j == 22)
                // {
                //     device_tensor_size = tensor_size;
                //     *(act_ptrs[j]) = device_act_iterator;
                // }
                // else
                // {
                device_tensor_size = (size_t)(tensor_size * ((float)model->devices[i].allocated_channels / model->config.channels));
                *(act_ptrs[j]) = device_act_iterator;
                //}
                device_act_iterator += device_tensor_size;
            }
        }
    }

 
    model->inputs = (int *)mallocCheck(B * T * sizeof(int));
    memcpy(model->inputs, inputs, B * T * sizeof(int));


    encoder_forward_sep(model, inputs, B, T);
    // print_encoded_sep(model, 3, 8);//检查bt位置上的encoded值

    float *residual;

    
    for (int l = 0; l < L; l++)
    {
        // **执行 LayerNorm**
        // layernorm_forward_sep_1(model, l, B, T);
        layernorm_forward_sep_1_modified(model, l, B, T);
        if (l == 0)
        {
            // print_ln1_sep(model, 0,3,8);
        }

        
        matmul_forward_sep_qkv(model, l, B, T, C, 3 * C);
        if (l == 0)
        {
            //print_qkv_sep(model, 0, 3,8, "Q");  //  Q 
            // print_qkv_sep(model, 0, 0, 0, "K");  
             //print_qkv_sep(model, 0, 3, 8, "V");  
        }

        // Self-Attention
        attention_forward_sep(model, l, B, T, C, NH);
        if (l == 0)
        {
            //print_atty_sep(model, 0, 3, 8);  //  layer 0, b=0, t=0 

        }

        // **执行 Attention Projection**
        matmul_forward_sep_attproj(model, l, B, T, C, C);
        if (l == 0)
        {
         //print_attproj_sep(model, 0, 3, 8);  //  layer 0, b=0, t=0  attproj 

        }

        
        residual_forward_sep_attproj(model, l, B, T, C);
        if (l == 0)
        {
           // print_residual2_sep(model, 0, 3, 8);  // layer 0, b=0, t=0  residual2 


        }

        //layernorm_forward_sep_2(model, l, B, T);
        layernorm_forward_sep_2_modified(model, l, B, T);
        if (l == 0)
        {
            //print_ln2_sep(model, 0, 3, 8);  // layer 0, b=0, t=0 ln2 
        }

        
        matmul_forward_sep_fc1(model, l, B, T, C, 4 * C);
        if (l == 0)
        {
            //print_fch_sep(model, 0, 3, 8);  //  layer 0, b=0, t=0  fch 
        
        }

        gelu_forward_sep(model, l, B, T);
        if (l == 0)
        {
            //print_fch_gelu_sep(model, 0, 3, 8);  

    
        }

        matmul_forward_sep_fc2(model, l, B, T, C, C);
        if (l == 0)
        {
           // print_fcproj_sep(model, 0, 3, 8);  

        }

        
        residual_forward_sep_fcproj(model, l, B, T, C);

        if (l == 11)
        {
            //print_residual3_sep(model, 11, 0, 0);  
        }
    }
    
    //layernorm_forward_sep_3(model, L, B, T);
    layernorm_forward_sep_3_modified(model, L, B, T);
    //print_lnf_sep(model, 3, 8);  //  batch=0, t=0 

    matmul_forward_sep_logits(model, B, T, C, Vp);
    //print_logits_sep(model, 3, 42);  //  batch=0, t=0  logits 

    //softmax_forward_sep(model, B, T, V, Vp);
    softmax_forward_sep_modified(model, B, T, V, Vp);
   // print_probs_sep(model, 3, 8);  //  batch=0, t=0  softmax 

}

void gpt2_free(GPT2_Sep *model)
{
    // release all device memories
    free(model->all_params_memory);
    free(model->all_acts_memory);
    free(model->inputs);

    
    for (int i = 0; i < 3; i++)
    {
        free(model->devices[i].device_params_memory);
        free(model->devices[i].device_acts_memory);
    }
}

#ifndef TESTING // if not defined as testing
// if we are TESTING (see test_gpt2.c), we'll skip the int main below
// ----------------------------------------------------------------------------
// sampler

unsigned int random_u32(uint64_t *state)
{
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(uint64_t *state)
{ // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample_mult(float *probabilities, int n, float coin)
{
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++)
    {
        cdf += probabilities[i];
        if (coin < cdf )
        {
            return i;
        }
    }
    // printf("Alert !only get coin cdf of %f\n", cdf);
    return n - 1; // in case of rounding errors
}




// ----------------------------------------------------------------------------


int main() {
    
    GPT2_Sep model;
    float load_factors[3] = {50.0f, 40.0f, 30.0f};
    gpt2_build_from_checkpoint_sep(&model, "gpt2_124M.bin", load_factors);

    
    Tokenizer tokenizer;
    tokenizer_init(&tokenizer, "gpt2_tokenizer.bin");

    
    int Vp=50304;
    int V=50257;
    int B = 4;        
    int T = 64;       
    int genT = 100;   
    int* inputs = (int*)mallocCheck(B * T * sizeof(int));

    for (int i = 0; i < B * T; i++) {
        inputs[i] = 50256; // 50256 是 EOT token
    }

    
    char input_text[150];
    printf("please input the text: ");
    fgets(input_text, sizeof(input_text), stdin);
    input_text[strcspn(input_text, "\n")] = '\0'; 

    
    int input_length = tokenizer_encode(&tokenizer, input_text, inputs, T);


    printf("input_length: %d\n", input_length);

    if (input_length < 0) {
        printf("Tokenizer error: input text too long or invalid.\n");
        return -1;
    }

    
    uint64_t rng_state = 1337; 
    printf("\nGenerating texts:\n---\n");

    float* global_probs = (float*)mallocCheck(model.config.padded_vocab_size * sizeof(float));
    memset(global_probs, 0, model.config.padded_vocab_size * sizeof(float));


    for (int t = input_length; t < genT; t++) {

        
        gpt2_forward_sep(&model, inputs, B, T);


        
        size_t offset = 0;
        // float coin_1 = random_f32(&rng_state);
        // int fake_t =(int)(coin_1*(float)B*(float)T);
        //printf("fake_t:%d\n",fake_t);
        for (int i = 0; i < 3; i++) {
            gpt_device *dev = &model.devices[i];
            int C_dev = dev->allocated_channels;
            size_t Vp_dev = (size_t)((float)model.config.padded_vocab_size * ((float)C_dev / model.config.channels));

            memcpy(global_probs + offset, dev->acts.probs + (t-1) * Vp_dev, Vp_dev * sizeof(float));
            //memcpy(global_probs + offset, dev->acts.probs + (int)(fake_t * (float)Vp_dev-1), Vp_dev * sizeof(float));
            offset += Vp_dev;
        }

        
        float coin = random_f32(&rng_state);
        //printf("coin: %.3f\n", coin);
        //vocab_size: 50257
        //padded_vocab_size: 50304
        int next_token = sample_mult(global_probs, model.config.vocab_size, coin);

        
        inputs[t] = next_token;

        
        if (tokenizer.init_ok) {
            const char* token_str = tokenizer_decode(&tokenizer, next_token);
            if (next_token != tokenizer.eot_token) {
                safe_printf(token_str);
            }
        } else {
            
            printf("%d ", next_token);
        }
        fflush(stdout);


    }

    printf("\n---\n");


    free(inputs);
    free(global_probs);
    gpt2_free(&model);
    tokenizer_free(&tokenizer);

    return 0;
}


#endif

