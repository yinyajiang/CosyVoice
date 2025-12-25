from cosyvoice.cli.vllm_cosvoice import AutoCosyVoice
import time


def main():
    cosyvoice = AutoCosyVoice(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', fp16=False, load_vllm=False, load_trt=False)
    
    prompt_text = '希望你以后能够做的比我还好呦。'
    prompt_wav = './asset/zero_shot_prompt.wav'
    tts_text = '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。'
    instruct = 'You are a helpful assistant. Imitate the tone and speaking style.'
    promptmodel = cosyvoice.get_promptmodel(f'{instruct}<|endofprompt|>{prompt_text}', prompt_wav)
    now = time.time()
    tts_generator = cosyvoice.inference_promptmodel(tts_text, promptmodel, stream=False)
    cosyvoice.save_tts_generator(tts_generator, 'test1.wav')
    print(f'get_promptmodel elapsed time: {time.time() - now:.2f}秒')

    promptmodel = cosyvoice.get_promptmodel_instruct(f'{instruct}<|endofprompt|>', prompt_wav)
    now = time.time()
    tts_generator = cosyvoice.inference_promptmodel(tts_text, promptmodel, stream=False)
    cosyvoice.save_tts_generator(tts_generator, 'test2.wav')
    print(f'get_promptmodel_instruct elapsed time: {time.time() - now:.2f}秒')


if __name__ == '__main__':
    main()
