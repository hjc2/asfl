
git add .


archive_dir="archive"
highest_version=$(ls -d $archive_dir/v* | sed 's/[^0-9]*//g' | sort -n | tail -n 1)

git commit -m "push for v$highest_version"

git push

echo "successfully done"
