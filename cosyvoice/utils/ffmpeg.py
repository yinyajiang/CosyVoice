import os
import shutil
import subprocess
import logging
import uuid


def merge_audio_files(file_path, files):
    if not files:
        raise ValueError('merge_audio_files: No audio files to merge')

    if isinstance(files, str):
        files = [files]

    dir_path = os.path.dirname(file_path)
    if not dir_path:
        dir_path = '.'

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if os.path.exists(file_path):
        os.remove(file_path)

    if len(files) == 1:
        src_ext = os.path.splitext(files[0])[1].lower()
        dst_ext = os.path.splitext(file_path)[1].lower()
        if src_ext == dst_ext:
            shutil.move(files[0], file_path)
        else:
            cmd = ['ffmpeg', '-y', '-i', files[0], file_path]
            _run_ffmpeg(cmd)
    else:
        list_file = os.path.join(os.path.dirname(os.path.abspath(files[0])), f'{uuid.uuid4().hex}_list.txt')
        try:
            with open(list_file, 'w') as f:
                for audio_file in files:
                    abs_path = os.path.abspath(audio_file)
                    f.write(f"file '{abs_path}'\n")

            cmd = [
                'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                '-i', list_file, file_path
            ]
            _run_ffmpeg(cmd)
        finally:
            if os.path.exists(list_file):
                os.remove(list_file)


def _run_ffmpeg(cmd):
    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
        return result
    except subprocess.CalledProcessError as e:
        logging.error(f'ffmpeg failed: {e.stderr.decode()}')
        raise
