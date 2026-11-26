"""Run with: streamlit run app.py"""
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.model import NOTICE, Predictor

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Smart Retinal Camera AI', page_icon='◉', layout='wide', initial_sidebar_state='collapsed')
st.title('Smart Retinal Camera AI')
st.caption('Image quality · Research classification · Visual explanation')
st.warning(NOTICE + ' Scores are uncalibrated and quality flags are heuristic.')


@st.cache_resource
def load_predictor(path, modified_ns, size):
    return Predictor(path, device='cpu')


with st.sidebar:
    st.header('Research model')
    checkpoints = sorted((ROOT / 'models').glob('*.pth')) + sorted((ROOT / 'runs').glob('*/best.pth'))
    choices = [str(p.relative_to(ROOT)) for p in checkpoints]
    chosen = st.selectbox('Checkpoint', choices) if choices else None
    st.caption('New training runs appear here after completion. The historical checkpoint stays unchanged.')
    show_cam = st.checkbox('Show Grad-CAM', value=True)
    st.caption('Images are processed in memory and are not saved by this app. Run locally for private data.')

demo_tab, evidence_tab = st.tabs(['Image demonstration', 'Recorded experiment'])
with evidence_tab:
    st.subheader('Historical APTOS baseline — one completed epoch')
    st.caption('These saved results describe the original held-out split. They are not performance measurements of another selected checkpoint or uploaded image.')
    st.dataframe(pd.read_csv(ROOT / 'results/test_metrics.csv'), hide_index=True)
    left, right = st.columns(2)
    left.image(str(ROOT / 'figures/test_confusion_matrix.png'))
    right.image(str(ROOT / 'figures/test_roc.png'))
    st.markdown('Image-level splits; one dataset; no external or clinical validation. See the research report for limitations.')

with demo_tab:
    uploaded = st.file_uploader('Upload an authorised retinal photograph', type=['png', 'jpg', 'jpeg'])
    if chosen is None:
        st.info('Add your trained .pth checkpoint to models/ or complete a training run first.')
    if uploaded is None:
        st.info('Choose an image to inspect its quality, model score and review indicators.')
    if uploaded is not None and chosen is not None:
        try:
            if uploaded.size > 20 * 1024 * 1024:
                raise ValueError('Please choose an image smaller than 20 MB.')
            image = Image.open(io.BytesIO(uploaded.getvalue()))
            if image.width * image.height > 40_000_000:
                raise ValueError('Please choose an image below 40 megapixels.')
            image = image.convert('RGB')
            if min(image.size) < 32:
                raise ValueError('The image is too small; use at least 32 pixels in each dimension.')
            model_path = ROOT / chosen
            stat = model_path.stat()
            predictor = load_predictor(str(model_path), stat.st_mtime_ns, stat.st_size)
            with st.spinner('Analysing image…'):
                result = predictor.predict(image)
            left, right = st.columns([1, 1])
            left.image(image, caption='Uploaded photograph')
            with right:
                st.subheader('Experimental model output')
                a, b = st.columns(2)
                a.metric('Model class', result['prediction'])
                b.metric('DR score', f"{result['dr_score']:.4f}")
                st.caption('Fixed decision threshold: 0.5. This score is not a calibrated disease probability.')
                if result['review_required']:
                    reasons = result['quality_flags'] + (['Score near decision threshold'] if result['uncertain_prediction'] else [])
                    st.warning('Experimental review flag: ' + '; '.join(reasons))
                else:
                    st.info('No heuristic flag triggered. This does not establish image adequacy or absence of disease.')
                st.dataframe(pd.DataFrame([result['quality']]), hide_index=True)
                st.download_button('Download research summary', json.dumps(result, indent=2), file_name='research_summary.json', mime='application/json')
                st.caption(f"Checkpoint SHA-256: {predictor.fingerprint[:16]}…")
            if show_cam:
                with st.spinner('Computing positive-class Grad-CAM…'):
                    overlay, nonzero = predictor.gradcam(image)
                st.image(overlay, caption='Positive DR-logit Grad-CAM overlay', width=350)
                st.caption('Exploratory explanation of the positive logit, including No-DR predictions. Not validated lesion localisation.')
                if not nonzero:
                    st.info('The positive-class activation map is flat for this image; do not interpret the overlay as localisation evidence.')
        except (UnidentifiedImageError, OSError, ValueError, RuntimeError, Image.DecompressionBombError) as error:
            st.error(f'Unable to analyse this image/checkpoint: {error}')
