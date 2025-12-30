from cosyvoice.cli.vllm_cosvoice import AutoCosyVoice
import time
import threading
from cosyvoice.utils.recommend import recommend_trt_concurrent
import io


def main():
    model = 'pretrained_models/Fun-CosyVoice3-0.5B'
    concurrent_num = recommend_trt_concurrent(model)
    cosyvoice = AutoCosyVoice(model_dir=model, fp16=False, load_vllm=False, load_trt=True, trt_concurrent=concurrent_num)
    
    instruct = 'You are a helpful assistant.'
    tts_text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
    prompt_text = '希望你以后能够做的比我还好呦。'
    prompt_wav = './asset/zero_shot_prompt.wav'

    tts_generator = cosyvoice.inference_instruct2(tts_text, f'{instruct}.模拟原始的音色和语调<|endofprompt|>', prompt_wav, stream=False)
    cosyvoice.save_tts_generator(tts_generator, 'test1.wav')
    return
   
    
    promptmodel = cosyvoice.get_promptmodel(f'{instruct}<|endofprompt|>{prompt_text}', prompt_wav)
    now = time.time()
    tts_generator = cosyvoice.inference_promptmodel(tts_text, promptmodel, stream=False)
    b = io.BytesIO()
    cosyvoice.save_tts_generator(tts_generator, b)
    with open('test1.wav', 'wb') as f:
        f.write(b.getvalue())
    print(f'elapsed time: {time.time() - now:.2f}秒')
    return

    with open('text.txt', 'r', encoding='utf-8') as f:
        tts_text = f.read()

    def case(thread_id):
        promptmodel = cosyvoice.get_promptmodel_instruct(f'{instruct}<|endofprompt|>', prompt_wav)
        now = time.time()
        tts_generator = cosyvoice.inference_promptmodel(tts_text, promptmodel, stream=False)
        cosyvoice.save_tts_generator(tts_generator, f'test2_{thread_id}.wav')
        print(f'Thread {thread_id} - get_promptmodel_instruct elapsed time: {time.time() - now:.2f}秒')

    threads = []
    for i in range(concurrent_num):
        thread = threading.Thread(target=case, args=(i,))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    print('所有线程执行完成')


if __name__ == '__main__':
    main()
