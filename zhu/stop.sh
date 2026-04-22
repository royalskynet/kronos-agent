#!/bin/bash
pid=$(pgrep -f "zhu_agent.py" || true)
if [ -z "$pid" ]; then
  echo "not running"
  exit 0
fi
kill "$pid" && echo "stopped pid=$pid"
