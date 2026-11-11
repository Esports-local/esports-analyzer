"""음성 합성(macOS say) 대사를 입힌 테스트 클립을 만든다. 호스트(맥)에서 실행, ffmpeg 는 vod-main 컨테이너.

구성 (60초, 1280x720, 25fps)
- 씬1 0~25초: t1.jpg 단체 사진 (전체 → 왼쪽 → 오른쪽 → 전체, 하드컷). 한국어 대화. 둘째 대사가 7초 컷을 넘는다.
- 25~27초: 암전 (fadeblack 2초)
- 씬2 27~50초: Tom Hanks 사진 (전체 → 근접 → 좌우반전, 하드컷). 일본어 ↔ 한국어 통역 대화.
- 50~51초: 디졸브
- 씬3 51~60초: mandelbrot (대사 없음)

python3 scripts/make_tts_clip.py   (맥 전용: say 명령 사용)  →  data/test/tts_drama.mp4
"""

import subprocess
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "data/test/tts_parts"
OUT = "data/test/tts_drama.mp4"

# (음성, 대사, 고정 시작 시각 또는 None=앞 대사 끝 + GAP)
SCENE1 = [
    ("Yuna", "통역사님, 내일 회의 자료는 다 보셨어요?", 0.8),
    ("Eddy (한국어(한국))", "네, 어젯밤에 전부 읽어 봤어요. 일본 쪽 요구가 꽤 까다롭더라고요.", 5.5),
    ("Yuna", "그래도 통역사님이 계시니까 걱정은 안 해요.", None),
    ("Eddy (한국어(한국))", "부담 주지 마세요. 저도 긴장돼요.", None),
    ("Yuna", "회의 끝나면 같이 저녁 먹을래요?", None),
]
SCENE2 = [
    ("Kyoko", "はじめまして。今日はよろしくお願いします。", 28.5),
    ("Eddy (한국어(한국))", "처음 뵙겠습니다. 오늘 잘 부탁드린다고 하십니다.", None),
    ("Kyoko", "契約の条件について、もう一度確認したいです。", None),
    ("Eddy (한국어(한국))", "계약 조건을 다시 한번 확인하고 싶다고 하십니다.", None),
    ("Yuna", "물론이죠. 천천히 말씀해 주세요.", None),
]
LIMITS = [(SCENE1, 24.8), (SCENE2, 49.8)]
GAP = 0.5
RATE = "190"


def tts(i: int, voice: str, text: str) -> tuple[Path, float]:
    path = TMP / f"line{i:02d}.wav"
    subprocess.run(["say", "-v", voice, "-r", RATE, "--data-format=LEI16@48000", "-o", str(path), text], check=True)
    with wave.open(str(path)) as w:
        return path, w.getnframes() / w.getframerate()


def main():
    TMP.mkdir(parents=True, exist_ok=True)
    placed = []  # (path, start, end, voice, text)
    i = 0
    for lines, limit in LIMITS:
        end = 0.0
        for voice, text, start in lines:
            i += 1
            path, dur = tts(i, voice, text)
            start = start if start is not None else end + GAP
            end = start + dur
            placed.append((path, start, end, voice, text))
            print(f"{start:6.2f}~{end:6.2f}  {voice.split()[0]:6s} {text}")
        assert end <= limit, f"씬 대사가 {limit}초를 넘음 ({end:.2f})"

    base = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2"
    norm = "fps=25,format=yuv420p,setsar=1,settb=AVTB,noise=alls=6:allf=t"
    # (입력 이미지, 길이, 앞단 필터)
    shots = [
        ("t1.jpg", 7, ""), ("t1.jpg", 6, "crop=iw/2:ih:0:0,"), ("t1.jpg", 6, "crop=iw/2:ih:iw/2:0,"), ("t1.jpg", 7, ""),
        ("Tom_Hanks_54745.png", 9, ""), ("Tom_Hanks_54745.png", 8, "crop=iw*0.6:ih*0.6:iw*0.2:ih*0.1,"),
        ("Tom_Hanks_54745.png", 9, "hflip,"),
    ]
    args, filt = [], []
    for k, (img, dur, pre) in enumerate(shots):
        args += ["-loop", "1", "-t", str(dur), "-i", f"$D/{img}"]
        filt.append(f"[{k}]{pre}{base},{norm}[s{k}]")
    n = len(shots)
    args += ["-f", "lavfi", "-t", "10", "-i", "mandelbrot=s=1280x720:r=25"]
    filt.append(f"[{n}]{norm}[s{n}]")
    filt.append("[s0][s1][s2][s3]concat=n=4:v=1[p1]")
    filt.append("[s4][s5][s6]concat=n=3:v=1[p2]")
    filt.append("[p1][p2]xfade=transition=fadeblack:duration=2:offset=25[p12]")
    filt.append(f"[p12][s{n}]xfade=transition=fade:duration=1:offset=50[v]")

    a0 = n + 1
    for j, (path, start, *_rest) in enumerate(placed):
        args += ["-i", f"/app/{path.relative_to(ROOT)}"]
        ms = int(start * 1000)
        filt.append(f"[{a0 + j}]aformat=sample_rates=48000:channel_layouts=stereo,adelay={ms}|{ms}[a{j}]")
    labels = "".join(f"[a{j}]" for j in range(len(placed)))
    filt.append(f"{labels}amix=inputs={len(placed)}:normalize=0,apad,atrim=0:60[a]")

    cmd = ("D=$(python -c 'import insightface,os;print(os.path.dirname(insightface.__file__))')/data/images; "
           "ffmpeg -v error -y " + " ".join(f'"{a}"' if " " in a or "$" in a or "=" in a else a for a in args)
           + f' -filter_complex "{";".join(filt)}" -map "[v]" -map "[a]" -t 60'
           f" -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a aac -b:a 128k {OUT}")
    subprocess.run(["docker", "run", "--rm", "-v", f"{ROOT}:/app", "-w", "/app", "vod-main", "sh", "-c", cmd], check=True)
    print("made", OUT)


if __name__ == "__main__":
    main()
