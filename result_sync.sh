while true; do 
  echo 'syncing'
  rsync -avz $1 /result
  sleep 10m
done
