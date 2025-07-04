/*
Defines the GPT-2 Tokenizer.
Only supports decoding, i.e.: tokens (integers) -> strings
This is all we need for unconditional generation.
If we wanted to later prompt the model, we'd have to add decoding.
Which could be tricky in C because of the regex involved, to look into later.
*/

#include <stdint.h>
#include <ctype.h>
#include <assert.h>
// our own utilities
// defines fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
#include "utils.h"

// ----------------------------------------------------------------------------

typedef struct {
    uint32_t vocab_size;
    char **token_table;
    int init_ok;
    int eot_token; // <|endoftext|> token id
} Tokenizer;

void safe_printf(const char *piece) {
    // the tokens are raw bytes, and we we only want to print the printable ones
    // many bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    // handle individual byte tokens
    // every token is asserted to be at least one byte so doing piece[1] is ok
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // weird byte, don't print it
        }
    }
    printf("%s", piece);
}

void tokenizer_init(Tokenizer *tokenizer, const char *filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        // try to be more helpful as we just added this feature, erase later
        printf("---\n");
        printf("WARNING: Failed to open the tokenizer file %s\n", filename);
        printf("The Tokenizer is a new feature added April 14 2024.\n");
        printf("Re-run `python train_gpt2.py` to write it\n");
        printf("---\n");
        tokenizer->init_ok = 0;
        return;
    }
    // read in the header
    uint32_t header[256];
    freadCheck(header, sizeof(uint32_t), 256, file);
    assert(header[0] == 20240328);
    int version = header[1];
    tokenizer->vocab_size = header[2];
    if (version == 1) {
        // version 1 didn't include the EOT token id
        // so we assume it is 50256, the EOT in GPT-2
        assert(tokenizer->vocab_size == 50257); // let's be defensive here
        tokenizer->eot_token = 50256;
    } else if (version == 2) {
        tokenizer->eot_token = header[3];
    } else {
        fprintf(stderr, "Tokenizer model file %s has bad version: %d\n", filename, version);
        exit(EXIT_FAILURE);
    }
    // read in all the tokens
    unsigned char length;
    tokenizer->token_table = (char **)mallocCheck(tokenizer->vocab_size * sizeof(char *));
    for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
        freadCheck(&length, sizeof(unsigned char), 1, file);
        assert(length > 0); // every token should be at least one character
        char *token_bytes = (char *)mallocCheck(length + 1);
        freadCheck(token_bytes, sizeof(char), length, file);
        token_bytes[length] = '\0';  // Add null terminator for printing
        tokenizer->token_table[i] = token_bytes;
    }
    // cleanups
    fcloseCheck(file);
    tokenizer->init_ok = 1;
}

const char *tokenizer_decode(Tokenizer *tokenizer, uint32_t token_id) {
    if (tokenizer->init_ok == 0) {
        return NULL;
    }
    if (token_id < tokenizer->vocab_size) {
        return tokenizer->token_table[token_id];
    } else {
        printf("invalid token id %u!\n", token_id);
        return NULL;
    }
}

//自己写了一个tokenizer的函数
int tokenizer_encode(Tokenizer *tokenizer, const char *text, uint32_t *output_ids, int max_tokens) {
    if (tokenizer->init_ok == 0) {
        printf("Tokenizer is not initialized!\n");
        return -1; // 返回错误
    }

    int token_count = 0;
    int text_len = strlen(text);
    int pos = 0;

    while (pos < text_len && token_count < max_tokens) {
        int best_match_len = 0;
        int best_match_id = -1;

        // **遍历 Token 表，查找最长匹配**
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            const char *token = tokenizer->token_table[i];
            int token_len = strlen(token);

            if (strncmp(text + pos, token, token_len) == 0) {
                if (token_len > best_match_len) {
                    best_match_len = token_len;
                    best_match_id = i;
                }
            }
        }

        // **匹配 Token 处理**
        if (best_match_id != -1) {
            output_ids[token_count++] = best_match_id;
            pos += best_match_len; // **跳过匹配 Token 的长度**
        } else {
            // **如果没有匹配 Token，降级按字符编码**
            unsigned char single_char = text[pos];
            if (single_char < tokenizer->vocab_size) {
                output_ids[token_count++] = single_char; // **使用 ASCII 值**
            } else {
                output_ids[token_count++] = tokenizer->eot_token; // **回退到 EOT token**
            }
            pos++;
        }
    }

    return token_count; // 返回编码后的 token 数量（即 `input_length`）
}



void tokenizer_free(Tokenizer *tokenizer) {
    if (tokenizer->init_ok) {
        for (uint32_t i = 0; i < tokenizer->vocab_size; i++) {
            free(tokenizer->token_table[i]);
        }
        free(tokenizer->token_table);
    }
}
