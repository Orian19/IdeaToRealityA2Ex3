from deepface import DeepFace
import gradio as gr
import os

# structured of dataset is: dataset/celeb_name/*.jpg (absolute path is needed)
DATASET_PATH = r'C:\IdeaToReality\A2\IdeaToRealityA2Ex3\gradio_celeb_face_recognition\celebs'


def find_similar_celebrity(image_path):
    """
    finding to whom the given person image is most similar to from the celebs
    :param image_path: jpg image of a person's face
    :return: name of the celeb (+distance which corresponds to similarity)
    """
    try:
        result = DeepFace.find(img_path=image_path, db_path=DATASET_PATH, enforce_detection=False)[0]
        # The result contains a pandas DataFrame with the paths of the similar images and their similarity scores
        if not result.empty:
            closest_image = result.iloc[0]
            celeb_name = os.path.basename(os.path.dirname(closest_image['identity']))
            return f"Most similar to: {celeb_name}, similarity(distance): {closest_image['distance']:.3f}"
    except Exception as e:
        return f"Error during processing: {str(e)}"


def main():
    iface = gr.Interface(fn=find_similar_celebrity,
                         inputs=gr.Image(),
                         outputs="text",
                         title="Find Your Celebrity Look-alike")

    iface.launch(share=True)


if __name__ == '__main__':
    main()
