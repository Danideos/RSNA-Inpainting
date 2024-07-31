from inpaint import inpaint_square, inpaint_grid
from app.utils.metric_utils import calculate_square_metrics, calculate_grid_metrics


def handle_inpaint_toggle_buttons(st, image_path, image, square, mask, img_size, inpaint_parameters, masks, offset, middle):
    with middle:
        with st.expander("Inpainting Options"):
            col1, col2, col3, col4 = st.columns([1,1,1,1])
            with col1:
                if st.button('Inpaint Square'):
                    inpaint_square(st, image_path, square, mask, img_size, inpaint_parameters=inpaint_parameters)
                    calculate_square_metrics(st, image, image_path, square)
                    st.rerun()

            with col2:
                if st.button('Inpaint Grid'):
                    inpaint_grid(st, image_path, img_size, masks, square, offset, inpaint_parameters=inpaint_parameters)
                    calculate_grid_metrics(st)
                    st.rerun()

            with col3:
                if st.button('Toggle Inpainted Square'):
                    st.session_state['show_inpainted_square'] = not st.session_state['show_inpainted_square']
            
            