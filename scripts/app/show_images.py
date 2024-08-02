from app.utils.general_utils import is_square_inpainted
from app.mask import overlay_mask
from app.utils.general_utils import get_keys
import matplotlib.pyplot as plt
from io import BytesIO

import streamlit as st


def generate_histogram_plot(original_histogram_values, inpainted_histogram_values):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(original_histogram_values, bins=80, range=(0, 80), alpha=0.5, label='Original')
    ax.hist(inpainted_histogram_values, bins=80, range=(0, 80), alpha=0.5, label='Inpainted')
    ax.set_xlabel('Value', size=14)
    ax.set_ylabel('Count', size=14)
    ax.set_title('Overlapping Histogram')
    ax.legend(loc='upper right')

    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    plt.close(fig)

    return buf

def show_image(image, square, mask, offset, middle_col, right_col):
    grid_key, square_key = get_keys(square, offset)
    x, y, square_length = square
    is_inpainted = is_square_inpainted(grid_key, square_key)
    print("searched:", grid_key, square_key, "found", is_inpainted)
    index = None if not is_inpainted else st.session_state['all_inpainted_square_images'][grid_key][square_key]['index']

    with middle_col:
        # with st.container():
        # st.markdown('<div class="left-align">', unsafe_allow_html=True)
        # Display the original image with the inpainted square if toggled on
        image_copy = image.copy()
        add_info = f'\nInpainting toggled off'
        if is_inpainted and st.session_state['show_inpainted_square']:
            inpainted_square_image = st.session_state['all_inpainted_square_images'][grid_key][square_key]['inpainted_square_image'][index]
            image_copy.paste(inpainted_square_image, (x, y))
            add_info = '\nInpainting toggled on'
        overlay_image = overlay_mask(image_copy, mask, square, offset)
        st.image(overlay_image, caption=f'Square at X: {x // square_length}, Y: {y // square_length}, Index: {index},{add_info}', use_column_width=True)
        # st.markdown('</div>', unsafe_allow_html=True)


    with right_col:
        _, centre, _ = st.columns([1, 3, 1])
        # Display the inpainted square metrics if they are calculated
        with centre:    
            if is_inpainted:
                for i in range(len(st.session_state['all_inpainted_square_images'][grid_key][square_key]['metrics'])):
                    expanded = True if i == index else False
                    with st.expander(f"Metrics for inpainted index {i}", expanded=expanded):
                        metrics = st.session_state['all_inpainted_square_images'][grid_key][square_key]['metrics'][i]
                        # Generate the histogram plot if it doesn't exist
                        if metrics['histogram_image'] is None:
                            metrics['histogram_image'] = generate_histogram_plot(metrics['original_histogram_data'], metrics['inpainted_histogram_data'])
                        
                        st.image(metrics['histogram_image'], caption="Histogram Comparison")
                        st.write(f"LPIPS: {metrics['lpips']}")
                        st.write(f"Mean Difference: {metrics['mean_diff']}")
                        st.write(f"KL Divergence: {metrics['kl_div']}")
                 