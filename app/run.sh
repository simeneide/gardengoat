#!/bin/bash
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root  --NotebookApp.token='' --NotebookApp.password='' --no-browser
while :
do
  echo "Press <CTRL+C> to exit."
  sleep 1
done