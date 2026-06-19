# sdrecon
for sub in 1 2 5 7
do
    python eval_caption.py caption/comparison/sdrecon/sub0${sub}_decoded_caption.json \
        caption/images --references_json caption/fmri_cococap.json
done

# braincap
for sub in 1 2 5 7
do
    python eval_caption.py caption/comparison/braincap/sub0${sub}_decoded_caption.json \
        caption/images --references_json caption/fmri_cococap.json
done

# onellm
python eval_caption.py caption/comparison/onellm/sub01_decoded_caption.json \
        caption/images --references_json caption/fmri_cococap.json

# umbrae captioning
for sub in 1 2 5 7
do
    python eval_caption.py caption/comparison/umbrae/sub0${sub}_decoded_caption.json \
        caption/images --references_json caption/fmri_cococap.json
done

# umbrae grounding
for sub in 1 2 5 7
do
    python eval_bbox_rec.py --path_out "bbox/umbrae/sub0${sub}_dim1024"
done
