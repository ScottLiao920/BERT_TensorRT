#!/usr/bin/env python3
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import ctypes
import argparse
import numpy as np
import tokenization
import tensorrt as trt
import data_processing as dp
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='BERT QA Inference')
    parser.add_argument('-e', '--bert_engine', dest='bert_engine',
            help='Path to BERT TensorRT engine')
    parser.add_argument('-p', '--passage', nargs='*',
            help='Text for paragraph/passage for BERT QA',
            default='')
    parser.add_argument('-b', '--batch_size', default='1')
    parser.add_argument('-pf', '--passage-file',
            help='File containing input passage',
            default='')
#     parser.add_argument('-q', '--question', nargs='*',
#             help='Text for query/question for BERT QA',
#             default='')
#     parser.add_argument('-qf', '--question-file',
#             help='File containiner input question',
#             default='')
    parser.add_argument('-v', '--vocab-file',
            help='Path to file containing entire understandable vocab',
            default='./pre-trained_model/uncased_L-24_H-1024_A-16/vocab.txt')
    parser.add_argument('-s', '--sequence-length',
            help='The sequence length to use. Defaults to 128',
            default=128, type=int)
    args, _ = parser.parse_known_args()
    return args

if __name__ == '__main__':
    cold_start = time.time()
    args = parse_args()
    batch_size = int(args.batch_size)
    if not args.passage == '':
        paragraph_text = ' '.join(args.passage)
    elif not args.passage_file == '':
        with open(args.passage_file, 'r', encoding='utf-8') as fin:
            tmp = fin.readlines()
        data = []
        for i in range(1, len(tmp)):
            label, sent = tmp[i].split('\t')
            data.append([label, sent])
    else:
        paragraph_text = input("Paragraph: ")

#     question_text = None
#     if not args.question == '':
#         question_text = ' '.join(args.question)
#     elif not args.question_file == '':
#         f = open(args.question_file, 'r')
#         question_text = f.read()
    try:
        print("\nPassage: {}".format(paragraph_text))
        data = paragraph_text
    except:
        print("\ninput file is provided, length = {}".format(len(data)))

    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=True)
    # The maximum number of tokens for the question. Questions longer than this will be truncated to this length.
    max_query_length = 8
    # When splitting up a long document into chunks, how much stride to take between chunks.
    doc_stride = 8
    # The maximum total input sequence length after WordPiece tokenization.
    # Sequences longer than this will be truncated, and sequences shorter
    max_seq_length = args.sequence_length
    # Extract tokecs from the paragraph
#     doc_tokens = dp.convert_doc_tokens(paragraph_text)

#     def question_features(question):
#         # Extract features from the paragraph and question
#         return dp.convert_examples_to_features(doc_tokens, question, tokenizer, max_seq_length, doc_stride, max_query_length)

    # Import necessary plugins for BERT TensorRT
    ctypes.CDLL("libnvinfer_plugin.so", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL("/workspace/TensorRT/demo/BERT/build/libcommon.so", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL("/workspace/TensorRT/demo/BERT/build/libbert_plugins.so", mode=ctypes.RTLD_GLOBAL)

    # The first context created will use the 0th profile. A new context must be created
    # for each additional profile needed. Here, we only use batch size 1, thus we only need the first profile.
    with open(args.bert_engine, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime, \
        runtime.deserialize_cuda_engine(f.read()) as engine, engine.create_execution_context() as context:

        input_shape = (batch_size, max_seq_length)
        input_nbytes = trt.volume(input_shape) * trt.int32.itemsize

        # Allocate device memory for inputs.
        d_inputs = [cuda.mem_alloc(input_nbytes) for binding in range(3)]
        # Create a stream in which to copy inputs/outputs and run inference.
        stream = cuda.Stream()

        # Specify input shapes. These must be within the min/max bounds of the active profile (0th profile in this case)
        # Note that input shapes can be specified on a per-inference basis, but in this case, we only have a single shape.
        for binding in range(3):
            context.set_binding_shape(binding, input_shape)
        assert context.all_binding_shapes_specified

        # Allocate output buffer by querying the size from the context. This may be different for different input shapes.
        h_output = cuda.pagelocked_empty(tuple(context.get_binding_shape(3)), dtype=np.float32)
        d_output = cuda.mem_alloc(h_output.nbytes)
        print("\nRunning Inference...")
        ttl_time = 0
        correct = 0

        for step in range(len(data)//batch_size):
            eval_start_time = time.time()            
            input_ids = np.random.randn(batch_size, max_seq_length)
            segment_ids = np.random.randn(batch_size, max_seq_length)
            input_mask = np.random.randn(batch_size, max_seq_length)
            for i in range(batch_size):
                short_paragraph_text = data[step * batch_size + i][1]
                doc_tokens = dp.convert_doc_tokens(short_paragraph_text)
                try:
                    features = dp.convert_examples_to_features(
                        doc_tokens, '', tokenizer, max_seq_length, doc_stride, max_query_length
                    )
                    input_ids[i] = features['input_ids']
                    segment_ids[i] = features['segment_ids']
                    input_mask[i] = features['input_mask']
                except:
                    print(doc_tokens)
                    i -= 1
            buffer_time = time.time()

            # asynchronous execution
            # Copy inputs(np arrays) into cuda memory
            cuda.memcpy_htod_async(d_inputs[0], input_ids.astype(np.int32), stream)
            cuda.memcpy_htod_async(d_inputs[1], segment_ids.astype(np.int32), stream)
            cuda.memcpy_htod_async(d_inputs[2], input_mask.astype(np.int32), stream)
            # Run inference, inference result is stored in cuda memory
            context.execute_async_v2(bindings=[int(d_inp) for d_inp in d_inputs] + [int(d_output)],stream_handle=stream.handle)
            # Transfer predictions back from GPU
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            # Synchronize the stream
            stream.synchronize()

            eval_time_elapsed = time.time() - eval_start_time
            if step == 0:
                cold_start_time = time.time() - cold_start
            ttl_time += eval_time_elapsed
        eval_time_elapsed = ttl_time / (step + 1)
        print("-----------------------------")
        print("Running Inference in {:.3f} Batches/Sec".format(
            1.0/eval_time_elapsed
        ))
        print("Time using for one batch is {:3f} ms".format(eval_time_elapsed*1000))
        print("Average time using for one inference is {:3f} ms".format(eval_time_elapsed*1000/batch_size))
        print("Time using for one batch cold start is {:.3f} ms".format(cold_start_time*1000))
        print("-----------------------------")
