"""Comparación controlada: mismo modelo/opciones, solo cambia el remuestreo."""
import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np
from faster_whisper import WhisperModel
from scipy.io import wavfile
try:
    from src.language.transcribe import prepare_audio
except ModuleNotFoundError:
    from transcribe import prepare_audio


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--sample',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model',default='small')
    args=parser.parse_args()
    model=WhisperModel(args.model,device='cpu',compute_type='int8')
    rows=[]
    with args.sample.open() as handle:
        files=[r['file_name'] for r in csv.DictReader(handle)]
    for name in files:
        path=args.root/'audio'/name
        sr,raw=wavfile.read(path)
        assert sr==8000 and raw.ndim==2 and raw.dtype==np.int16
        old=raw[:,0].astype(np.float32)/32768.0
        corrected,_,_=prepare_audio(path)
        row={'file_name':name,'model':args.model}
        for label,audio in [('without_resampling',old),('with_resampling',corrected)]:
            try:
                segments,_=model.transcribe(audio,language='es',beam_size=1)
                text=' '.join(s.text.strip() for s in segments).strip()
                row[label+'_text']=text
                row[label+'_words']=len(re.findall(r'\b\w+\b',text))
                row[label+'_status']='success' if text else 'empty'
                row[label+'_error']=''
            except Exception as exc:
                row[label+'_text']=''
                row[label+'_words']=0
                row[label+'_status']='error'
                row[label+'_error']=f'{type(exc).__name__}: {exc}'
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if not k.endswith('_text')},ensure_ascii=False),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
