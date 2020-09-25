mkdir upload
FILES=$(ls data | grep tub)
for FILE in $FILES
do
  echo "TARRING:"
  tar -zcvf "upload/$FILE.tar.gz" "data/$FILE/"
done

echo "COPY..:"
rclone move upload/ goatdrive:gardengoat/data -P
echo "Done transferring."