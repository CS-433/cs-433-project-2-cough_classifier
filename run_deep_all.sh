for segmentation_type in coarse fine no; do
  for drop_user_features in False True; do
    for split_by_expert in True False; do
      python run_deep.py $segmentation_type $drop_user_features $split_by_expert True
    done
  done
done
