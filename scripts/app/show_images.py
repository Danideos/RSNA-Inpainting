from PIL import Image
from mask import overlay_mask


def show_image(st, image, square, mask, middle_col, right_col):
    with middle_col:
        x, y, size = square
        # Display the original image with the inpainted square if toggled on
        image_copy = image.copy()
        add_info = f'\nInpainting toggled off'
        print("searching for key:", square)
        if square in st.session_state['all_inpainted_square_images'] and st.session_state['show_inpainted_square']:
            inpainted_square_image = st.session_state['all_inpainted_square_images'][square]
            image_copy.paste(inpainted_square_image, (y, x))
            add_info = '\nInpainting toggled on'
        overlay_image = overlay_mask(image_copy, mask, square, st.session_state['show_selection'], st.session_state['show_correct_grid'])
        st.image(overlay_image, caption=f'Square at X: {x // size}, Y: {y // size}{add_info}', use_column_width=True)

    with right_col:
        # Display the inpainted square metrics if they are calculated
        key = ((x, y), size)
        if key in st.session_state and len(st.session_state[key]) > 0:
            metrics_index = st.session_state.metrics_index.get(key, 0)
            metrics = st.session_state[key][metrics_index]
            st.image(metrics['histogram_image'], caption="Histogram Comparison")
            st.write(f"LPIPS: {metrics['lpips']}")
            st.write(f"Mean Difference: {metrics['mean_diff']}")
            st.write(f"KL Divergence: {metrics['kl_div']}")