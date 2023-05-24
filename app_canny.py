# This file is adapted from https://github.com/lllyasviel/ControlNet/blob/f4748e3630d8141d7765e2bd9b1e348f47847707/gradio_canny2image.py
# The original license file is LICENSE.ControlNet in this repo.
import gradio as gr


def create_demo(process, max_images=12, default_num_images=3):
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown('## Control Stable Diffusion with Canny Edge Maps')
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type='numpy')
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button(label='Run')
                with gr.Accordion('Advanced options', open=False):
                    num_samples = gr.Slider(label='Images',
                                            minimum=1,
                                            maximum=max_images,
                                            value=default_num_images,
                                            step=1)
                    image_resolution = gr.Slider(label='Image Resolution',
                                                 minimum=256,
                                                 maximum=512,
                                                 value=512,
                                                 step=256)
                    canny_low_threshold = gr.Slider(
                        label='Canny low threshold',
                        minimum=1,
                        maximum=255,
                        value=100,
                        step=1)
                    canny_high_threshold = gr.Slider(
                        label='Canny high threshold',
                        minimum=1,
                        maximum=255,
                        value=200,
                        step=1)
                    num_steps = gr.Slider(label='Steps',
                                          minimum=1,
                                          maximum=100,
                                          value=20,
                                          step=1)
                    guidance_scale = gr.Slider(label='Guidance Scale',
                                               minimum=0.1,
                                               maximum=30.0,
                                               value=9.0,
                                               step=0.1)
                    seed = gr.Slider(label='Seed',
                                     minimum=-1,
                                     maximum=2147483647,
                                     step=1,
                                     randomize=True)
                    a_prompt = gr.Textbox(
                        label='Added Prompt',
                        value='best quality, extremely detailed')
                    n_prompt = gr.Textbox(
                        label='Negative Prompt',
                        value=
                        'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
                    )
            with gr.Column():
                result = gr.Gallery(label='Output',
                                    show_label=False,
                                    elem_id='gallery').style(grid=2,
                                                             height='auto')
        inputs = [
            input_image,
            prompt,
            a_prompt,
            n_prompt,
            num_samples,
            image_resolution,
            num_steps,
            guidance_scale,
            seed,
            canny_low_threshold,
            canny_high_threshold,
        ]
        prompt.submit(fn=process, inputs=inputs, outputs=result)
        run_button.click(fn=process,
                         inputs=inputs,
                         outputs=result,
                         api_name='canny')
    return demo


if __name__ == '__main__':
    from model import Model
    model = Model()
    demo = create_demo(model.process_canny)
    demo.queue().launch()