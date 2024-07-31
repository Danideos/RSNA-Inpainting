from app.utils.general_utils import is_square_inpainted
from app.mask import overlay_mask
from app.utils.general_utils import get_keys

import streamlit as st


def show_image(image, square, mask, offset, middle_col, right_col):
    grid_key, square_key = get_keys(square, offset)
    x, y, square_length = square
    is_inpainted = is_square_inpainted(grid_key, square_key)
    index = None if not is_inpainted else st.session_state['all_inpainted_square_images'][grid_key][square_key]['index']

    with middle_col:
        # with st.container():
        # st.markdown('<div class="left-align">', unsafe_allow_html=True)
        # Display the original image with the inpainted square if toggled on
        image_copy = image.copy()
        add_info = f'\nInpainting toggled off'
        if is_inpainted and st.session_state['show_inpainted_square']:
            inpainted_square_image = st.session_state['all_inpainted_square_images'][grid_key][square_key]['inpainted_square_image'][index]
            image_copy.paste(inpainted_square_image, (y, x))
            add_info = '\nInpainting toggled on'
        overlay_image = overlay_mask(image_copy, mask, square, offset, st.session_state['show_selection'], st.session_state['show_correct_grid'])
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
                        st.image(metrics['histogram_image'], caption="Histogram Comparison")
                        # st.write(f"LPIPS: {metrics['lpips']}")
                        st.write(f"Mean Difference: {metrics['mean_diff']}")
                        st.write(f"KL Divergence: {metrics['kl_div']}")
                 