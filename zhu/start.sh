#!/bin/bash
# Zhu Agent — manual start (nohup background).
# Stop via: ./stop.sh  (or pkill -f zhu_agent.py)
set -e
cd "$(dirname "$0")"
if pgrep -f "zhu_agent.py" >/dev/null; then
  echo "already running (pid $(pgrep -f zhu_agent.py))"
  exit 0
fi
nohup python3 zhu_agent.py >> zhu_agent.out 2>&1 &
sleep 1
if pgrep -f "zhu_agent.py" >/dev/null; then
  echo "started pid=$(pgrep -f zhu_agent.py) mode=$(python3 -c 'import json;print("LIVE" if json.load(open("zhu_config.json"))["live"] else "DRY")')"
  echo "log: tail -f $(pwd)/zhu_agent.log"
else
  echo "start failed — see zhu_agent.out"
  tail -30 zhu_agent.out
  exit 1
fi
