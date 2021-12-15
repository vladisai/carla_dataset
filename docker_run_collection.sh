#!/bin/bash

TMP_DIR=/tmp/result_${NGC_JOB_ID}/
mkdir $TMP_DIR
chmod 777 $TMP_DIR

echo 'Running syncer'
./result_sync.sh $TMP_DIR &
SYNCER=$!

SDL_VIDEODRIVER=offscreen /home/carla/CarlaUE4.sh -nosound -quality-level=Epic -fps=10 -opengl -benchmark &
echo 'Sleeping for 2 mins to let server start'
sleep 2m 

echo 'Running main script'

while true; do
	python bbox.py -d $TMP_DIR "$@" &
	COLLECTION_PID=$!
	sleep 2h
	kill $COLLECTION_PID
done

echo 'Main script is over, killing syncer'

kill $SYNCER

echo 'Copying results once again'

rsync -avz $TMP_DIR /result
