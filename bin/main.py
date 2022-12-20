import argparse, os, sys, json
import math
from threading import Thread
from sentence_analyzer import SentenceAnalyzer

import warnings
warnings.filterwarnings(action='ignore')
import torch

def analyze(sentences, batch_size, output_file, device=None):
    print(f"gpu {device} 시작")
    t = SentenceAnalyzer(batch_size=batch_size, device=device)
    res_morphology = t.morphology_analysis(sentences)
    res_parsing = t.dependency_parsing(res_morphology["result_sentences"], file=output_file)

    fail_count = res_morphology["error_number"] + res_parsing["error_number"] + len(sentences) - len(res_parsing['result_sentences'])
    print(f"\n gpu {device} 분석 실패: {fail_count}")
    print(f"gpu {device} 끝")


if __name__ == "__main__":
    # python main.py -root_dir ../testset -input_file test.txt -batch_size 30 -save_file result.txt -use_gpu
    parser_main = argparse.ArgumentParser(description="main")
    parser_main.add_argument('-input_file', type=str, required=True)
    parser_main.add_argument('-save_file', type=str, default="result.txt")
    parser_main.add_argument('-use_gpu', nargs="+",type=int, default=None)
    parser_main.add_argument('-batch_size', type=int, default=30)

    opt = parser_main.parse_args()
    input_file = opt.input_file
    save_file = opt.save_file
    gpu_list = opt.use_gpu
    batch_size = opt.batch_size
    sentences = []
    try:
        filename, file_extension = os.path.splitext(input_file)
        if file_extension == '.json':
            with open(input_file, 'r') as f:
                sentences = json.load(f)
        else:
            with open(os.path.join(input_file), 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    sentences.append(line)
        # random.shuffle(sentences)
    except UnicodeDecodeError:
        print("File Encoding 오류: 기본 인코딩은 utf-8입니다.")
        exit()
    sentences = sentences[:1]
    sentence_size = len(sentences)
    print("문장 수: {}".format(sentence_size))
    if gpu_list and len(gpu_list) >= 2:
        available_gpu_list = []
        for gpu_id in gpu_list:
            try:
                torch.cuda.get_device_name(gpu_id)
                available_gpu_list.append(gpu_id)
            except AssertionError:
                continue
        gpu_list = available_gpu_list
            
        thread_list = []
        thread_sent_size = max(batch_size, math.ceil(sentence_size / len(gpu_list)))
        thread_output_file_list = []
        for i, thread_gpu in enumerate(gpu_list):
            thread_sents = sentences[thread_sent_size*i:thread_sent_size*(i+1)]
            if not thread_sents:
                continue
            thread_sents.sort(key=lambda x: len(x))
            thread_output_file = save_file + '_' + str(i)
            thread = Thread(target=analyze, args=(thread_sents, batch_size, thread_output_file, thread_gpu))
            thread_list.append(thread)
            thread_output_file_list.append(thread_output_file)
        
        for thread in thread_list:
            thread.start()
 
        for thread in thread_list:
            thread.join()
        
        with open(save_file, "w") as outfile:
            for filename in thread_output_file_list:
                with open(filename) as infile:
                    contents = infile.read()
                    outfile.write(contents)
    else:
        if gpu_list:
            device = 0
        else:
            device = None
        t = SentenceAnalyzer(batch_size=batch_size, device=device)
        res_morphology = t.morphology_analysis(sentences)
        res_parsing = t.dependency_parsing(res_morphology["result_sentences"], file=save_file)

        fail_count = res_morphology["error_number"] + res_parsing["error_number"] + sentence_size - len(res_parsing['result_sentences'])
        print("\n분석 실패: {}".format(fail_count))
        print("끝")
