import os, json, subprocess, glob, sys
sys.path.insert(0,"C:/Users/amass/tennis_analysis")
from dotenv import load_dotenv
load_dotenv("C:/Users/amass/tennis_analysis/.env")
import boto3
acct=os.environ["CF_ACCOUNT_ID"]
s3=boto3.client("s3", endpoint_url=f"https://{acct}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["CF_R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["CF_R2_SECRET_ACCESS_KEY"], region_name="auto")
PY="C:/Users/amass/tennis_analysis/venv/Scripts/python.exe"
wins=glob.glob("C:/Users/amass/balltrack/windows/strike_*.mp4")
print("windows:",len(wins))
for w in wins:
    short=os.path.basename(w)[len("strike_"):-len(".mp4")]
    od=f"C:/Users/amass/balltrack/seed_{short}"
    r=subprocess.run([PY,"C:/Users/amass/balltrack/track_ball.py",
        "--model_path","C:/Users/amass/balltrack/weights/tracknet.pt",
        "--video_path",w,"--out_dir",od], capture_output=True,text=True)
    tj=os.path.join(od,"ball_track.json")
    if not os.path.exists(tj):
        print(f"{short}: FAILED {r.stderr.strip()[-160:]}"); continue
    d=json.load(open(tj))
    # normalize coords to 0..1 by frame size for resolution-independent labeling
    W=d.get("summary",{}).get("width",1280); H=d.get("summary",{}).get("height",720)
    seed={"clip":f"strike_{short}","fps":d.get("summary",{}).get("fps",120),
          "width":W,"height":H,
          "track":[{"frame":t["frame"],
                    "x":round(t["x"]/W,5) if t.get("x") is not None else None,
                    "y":round(t["y"]/H,5) if t.get("y") is not None else None,
                    "visible":bool(t.get("visible")) and t.get("x") is not None} for t in d.get("track",[])]}
    s3.put_object(Bucket="tennis-videos",Key=f"uploads/ball_seed_{short}.json",
        Body=json.dumps(seed).encode(),ContentType="application/json")
    n=sum(1 for t in seed["track"] if t["visible"])
    print(f"{short}: seed uploaded ({n}/{len(seed['track'])} visible)")
print("done")
