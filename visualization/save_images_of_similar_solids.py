from occwl.solid import Solid
from occwl.viewer import Viewer

class SimilarSolidImageSaver:
    def __init__(self, solid):
        self.solid = solid
        self.viewer = Viewer(
            size = (400, 400),
            axes = False,
            background_gradient_color1 = [255, 255, 255],
            background_gradient_color2 = [255, 255, 255]
        )

    def save_image_of_selected_faces(
            self,
            pathname,
            values_for_faces, 
            threshold
        ):
        for index, face in enumerate(self.solid.faces()):
            if values_for_faces[index] < threshold:
                color = "red"
            else:
                color = None
            self.viewer.display(face, color=color)

        self.viewer.show()
        self.viewer._display.FitAll()
        self.viewer.save_image(pathname)