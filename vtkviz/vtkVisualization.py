import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from abc import ABC

# This is just that y and z may be inverted depending on what you are used to.
dir_x = np.array([1, 0, 0])
dir_y = np.array([0, 0, 1])
dir_z = np.array([0, 1, 0])

def center_mesh_give_size(vertices):
    prod = np.inner(vertices, dir_z)
    p_max = np.amax(prod)
    p_min = np.amin(prod)
    p1 = p_max + 0.1*(p_max-p_min)
    p2 = p_min + 0.1*(p_min-p_max)
    return p1, p2


def gen_tex_coords(vertices, axes=(0, 1)):
    """
    Ref:
    - https://github.com/llorz/SGA19_zoomOut/blob/master/utils/%2BMESH/%2BMESH_IO/generate_tex_coords.m

    Args:
        vertices (np.ndarray): (V, 3)
        axes (tuple): plane of axes[0] and axes[1]

    Returns:
        np.ndarray: (V, 2)
    """
    vt = vertices[:, axes]  # (V, 2)
    vt = vt - np.amin(vt, axis=0, keepdims=True)
    vt = vt / np.amax(vt)
    # Offset
    vt = ((vt * 2 - 1) * 0.9 + 1) / 2
    return vt

def ReadPolyData(file_name):
    import os
    path, extension = os.path.splitext(file_name)
    extension = extension.lower()
    if extension == ".ply":
        reader = vtk.vtkPLYReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".vtp":
        reader = vtk.vtkXMLpoly_dataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".obj":
        reader = vtk.vtkOBJReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".stl":
        reader = vtk.vtkSTLReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    elif extension == ".g":
        reader = vtk.vtkBYUReader()
        reader.SetGeometryFileName(file_name)
        reader.Update()
        poly_data = reader.GetOutput()
    else:
        # Return a None if the extension is unknown.
        poly_data = None
    return poly_data


class VTKEntity3D(ABC):
    def __init__(self, mapper: vtk.vtkMapper):
        self._actor = vtk.vtkActor()
        self._actor.SetMapper(mapper)

    @property
    def actor(self):
        return self._actor


class VTKFile(VTKEntity3D):
    def __init__(self, filename):
        self.polydata = ReadPolyData(filename)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.polydata)

        super().__init__(mapper)

    def update_file(self, filename):
        new_poly = ReadPolyData(filename)
        self.polydata.ShallowCopy(new_poly)
        self.polydata.Modified()

    def update_poly(self, poly):
        self.polydata.ShallowCopy(poly)
        self.polydata.Modified()

        
class VTKPoly(VTKEntity3D):
    def __init__(self, poly):
        self.polydata=poly
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        super().__init__(mapper)
    def update_poly(self, poly):
        self.polydata.ShallowCopy(poly)
        self.polydata.Modified()


class VTKPointCloud(VTKEntity3D):
    def __init__(self, points: np.ndarray=None, colors: np.ndarray=None):
        assert (points is None and colors is None) or (points is not None and colors is not None)

        self.num_points = 0

        # VTK geometry representation
        self._points = vtk.vtkPoints()

        # VTK color representation
        self._colors = vtk.vtkUnsignedCharArray()
        self._colors.SetName("Colors")
        self._colors.SetNumberOfComponents(3)

        # Visualization pipeline
        # - Data source
        point_data = vtk.vtkPolyData()
        point_data.SetPoints(self._points)
        point_data.GetPointData().SetScalars(self._colors)

        # - Automatically generate topology cells from points
        mask_points = vtk.vtkMaskPoints()
        mask_points.SetInputData(point_data)
        mask_points.GenerateVerticesOn()
        mask_points.SingleVertexPerCellOn()
        mask_points.Update()

        # - Map the data representation to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(mask_points.GetOutputPort())

        super().__init__(mapper)

        self.add_points(points, colors)

    def add_points(self, points: np.ndarray, colors: np.ndarray): #points, colors Nx3
        assert len(points.shape) == 2
        assert len(colors.shape) == 2
        assert points.shape[0] == colors.shape[0]
        assert points.shape[1] == 3
        assert colors.shape[1] == 3

        [num_new_points, _] = points.shape

        # Allocate additional memory
        self._points.Resize(self.num_points + num_new_points * 2)
        self._colors.Resize(self.num_points + num_new_points * 2)

        # Add points
        for point_idx in range(num_new_points):
            for _ in range(2):  # add every point twice due to vtk bug
                self._points.InsertNextPoint(points[point_idx, :])
                self._colors.InsertNextTuple(colors[point_idx, :])

        self._points.Modified()
        self._colors.Modified()

    def set_point_size(self, size: int):
        self.actor.GetProperty().SetPointSize(size)


class VTKHull(VTKEntity3D):
    def __init__(self, convex_hull, color=None):
        # - convex hull is the result of delaunay triangulation
        self.convex_hull = convex_hull
        # - Map the data representation to graphics primitives
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(self.convex_hull.GetOutput())

        super().__init__(mapper)
        if color is None:
            self.actor.GetProperty().SetColor(0.5, 0.5, 1.0)
        else:
            self.actor.GetProperty().SetColor(color[0], color[1], color[2])
        self.actor.GetProperty().SetOpacity(0.5)


class VTKVectorField(VTKEntity3D):
    def __init__(self, positions: np.ndarray, vectors: np.ndarray, color=[1, 0, 0]):
        self.num_vectors = 0

        # VTK position representation
        self._positions = vtk.vtkPoints()

        # VTK vector representation
        self._vectors = vtk.vtkFloatArray()
        self._vectors.SetName("Vector Field")
        self._vectors.SetNumberOfComponents(3)

        # VTK norm vector representation
        self._vectors_norm = vtk.vtkFloatArray()
        self._vectors_norm.SetName("Vector Field")


        # Visualization Pipeline
        # - Data source
        position_data = vtk.vtkPolyData()
        position_data.SetPoints(self._positions)
        position_data.GetPointData().AddArray(self._vectors)
        position_data.GetPointData().SetActiveVectors("Vector Field")

        # - Add the vector arrays as 3D Glyphs

        grid = vtk.vtkPolyData()
        grid.SetPoints(self._positions)
        grid.GetPointData().SetVectors(self._vectors)

        #grid.GetPointData().SetScalars(self._vectors_norm)

        arrow = vtk.vtkArrowSource()
        arrow.SetTipResolution(16)
        arrow.SetTipLength(0.3)
        arrow.SetTipRadius(0.1)

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(grid)
        glyph.SetSourceConnection(arrow.GetOutputPort())
        glyph.SetVectorModeToUseVector()
        #glyph.SetScaleModeToScaleByScalar()
        glyph.SetScaleModeToScaleByVector()
        glyph.SetColorModeToColorByScalar()
        glyph.OrientOn()

        glyph.Update()

        # - Map the data representation to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        #mapper.SetInputConnection(add_arrows.GetOutputPort())
        mapper.SetInputConnection(glyph.GetOutputPort())

        super().__init__(mapper)
        if color is not None:
            self.actor.GetProperty().SetColor(color[0], color[1], color[2])

        self.add_vectors(positions, vectors)

    def add_vectors(self, positions: np.ndarray, vectors: np.ndarray):
        assert len(positions.shape) == 2
        assert len(vectors.shape) == 2
        assert positions.shape[0] == vectors.shape[0]
        assert positions.shape[1] == 3
        assert vectors.shape[1] == 3

        [num_new_vectors, _] = vectors.shape

        # Allocate additional memory
        self._positions.Resize(self.num_vectors + num_new_vectors)
        self._vectors.Resize(self.num_vectors + num_new_vectors)
        self._vectors_norm.Resize(self.num_vectors + num_new_vectors)

        # Add points
        for vector_idx in range(num_new_vectors):
            self._positions.InsertNextPoint(positions[vector_idx, :])
            self._vectors.InsertNextTuple(vectors[vector_idx, :])
            self._vectors_norm.InsertNextValue(np.linalg.norm(vectors[vector_idx, :]))

        self._positions.Modified()
        self._vectors.Modified()
        self._vectors_norm.Modified()


class VTKSurface(VTKEntity3D):
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, color=None):
        self.num_vertices = 0
        self.num_faces = 0

        # VTK point cloud representation
        self._vertices = vtk.vtkPoints()

        # VTK polygone(surface) representation
        self._faces = vtk.vtkCellArray()

        # Visualization Pipeline
        # - Data source
        self.surface_data = vtk.vtkPolyData()
        self.surface_data.SetPoints(self._vertices)
        self.surface_data.SetPolys(self._faces)

        # - Add the vector arrays as 3D Glyphs


        # - Map the data representation to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.surface_data)

        super().__init__(mapper)
        if color is None:
            super().__init__(mapper)
            self.actor.GetProperty().SetColor(0.5, 0.5, 1.0)
        elif type(color)== list:
            super().__init__(mapper)
            self.actor.GetProperty().SetColor(color[0], color[1], color[2])
        elif type(color)==str:
            reader = vtk.vtkJPEGReader()
            reader.SetFileName(color)
            reader.Update()
            VT0 = gen_tex_coords(self.vertices)
            self.surface_data.GetPointData().SetTCoords(numpy_to_vtk(VT0))
            tu = vtk.vtkTexture()
            tu.SetInputData(reader.GetOutput())
            tu.SetInterpolate(False)
            self._actor.SetTexture(tu)
        elif len(color.shape) == 1:
            colors_array = vtk.vtkDoubleArray()
            for i in range(color.shape[0]):
                colors_array.InsertNextValue(color[i])
            self.surface_data.GetPointData().SetScalars(colors_array)
            self.lut = vtk.vtkLookupTable()
            range_data = self.surface_data.GetPointData().GetScalars().GetRange()
            self.lut.SetTableRange(range_data)
            self.lut.SetHueRange(0.1, 0.35)
            self.lut.SetSaturationRange(.8, 1.)
            self.lut.SetValueRange(0.7, 1.0)
            mapper.SetLookupTable(self.lut)
            mapper.SetScalarRange(range_data)
            mapper.Update()
            super().__init__(mapper)
        else:
            colors_array = vtk.vtkUnsignedCharArray()
            colors_array.SetNumberOfComponents(3)
            for i in range(color.shape[0]):
                colors_array.InsertTuple(i, color[i, :])
            self.surface_data.GetPointData().SetScalars(colors_array)
            #super.__init__(mapper)
        self.actor.GetProperty().SetOpacity(1)

        self.add_vectors(vertices, faces)

    def add_vectors(self, vertices: np.ndarray, faces: np.ndarray):
        assert len(vertices.shape) == 2
        assert len(faces.shape) == 2
        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3

        # Add points
        [num_vertices, _] = vertices.shape
        for vertex_idx in range(num_vertices):
            self._vertices.InsertNextPoint(vertices[vertex_idx, 0], vertices[vertex_idx, 1], vertices[vertex_idx, 2])
        [num_faces, _] = faces.shape
        for face_idx in range(num_faces):
            self._faces.InsertNextCell(3)
            for corner_idx in range(3):
                self._faces.InsertCellPoint(faces[face_idx, corner_idx])
        # Allocate additional memory
        self._vertices.Resize(self.num_vertices + num_vertices)
        #self._faces.Resize(self.num_faces + num_faces*3)

        self._vertices.Modified()
        self._faces.Modified()


    def change_points(self, vertices: np.ndarray, faces: np.ndarray):
        assert len(vertices.shape) == 2
        assert len(faces.shape) == 2
        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3

        # Add points
        verticesPoints = vtk.vtkPoints()
        facesCells = vtk.vtkCellArray()
        [num_vertices, _] = vertices.shape
        for vertex_idx in range(num_vertices):
            verticesPoints.InsertNextPoint(vertices[vertex_idx, 0], vertices[vertex_idx, 1], vertices[vertex_idx, 2])
        [num_faces, _] = faces.shape
        for face_idx in range(num_faces):
            facesCells.InsertNextCell(3)
            for corner_idx in range(3):
                facesCells.InsertCellPoint(faces[face_idx, corner_idx])
        # Allocate additional memory
        verticesPoints.Resize(self.num_vertices + num_vertices)
        # self._faces.Resize(self.num_faces + num_faces*3)

        self._vertices.ShallowCopy(verticesPoints)
        self._faces.ShallowCopy(facesCells)
        self._vertices.Modified()
        self._faces.Modified()


class VTKSurfaceP2P(VTKEntity3D):
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, VT0:np.ndarray, p2pmap:np.array, color=None):
        self.num_vertices = 0
        self.num_faces = 0
        self.VT0 = VT0
        # VTK point cloud representation
        self._vertices = vtk.vtkPoints()

        # VTK polygone(surface) representation
        self._faces = vtk.vtkCellArray()

        # Visualization Pipeline
        # - Data source
        self.surface_data = vtk.vtkPolyData()
        self.surface_data.SetPoints(self._vertices)
        self.surface_data.SetPolys(self._faces)

        # - Add the vector arrays as 3D Glyphs


        # - Map the data representation to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(self.surface_data)

        super().__init__(mapper)
        if color is None:
            super().__init__(mapper)
            self.actor.GetProperty().SetColor(0.5, 0.5, 1.0)
        elif type(color) == list:
            super().__init__(mapper)
            self.actor.GetProperty().SetColor(color[0], color[1], color[2])
        elif type(color) == str:
            reader = vtk.vtkJPEGReader()
            reader.SetFileName(color)
            reader.Update()
            self.surface_data.GetPointData().SetTCoords(numpy_to_vtk(self.VT0[p2pmap]))
            tu = vtk.vtkTexture()
            tu.SetInputData(reader.GetOutput())
            tu.SetInterpolate(False)
            self._actor.SetTexture(tu)
        elif len(color.shape) == 1:
            colors_array = vtk.vtkDoubleArray()
            for i in range(color.shape[0]):
                colors_array.InsertNextValue(color[i])
            self.surface_data.GetPointData().SetScalars(colors_array)
            self.lut = vtk.vtkLookupTable()
            range_data = self.surface_data.GetPointData().GetScalars().GetRange()
            self.lut.SetTableRange(range_data)
            self.lut.SetHueRange(0.1, 0.35)
            self.lut.SetSaturationRange(.8, 1.)
            self.lut.SetValueRange(0.7, 1.0)
            mapper.SetLookupTable(self.lut)
            mapper.SetScalarRange(range_data)
            mapper.Update()
            super().__init__(mapper)
        else:
            colors_array = vtk.vtkUnsignedCharArray()
            colors_array.SetNumberOfComponents(3)
            for i in range(color.shape[0]):
                colors_array.InsertTuple(i, color[i, :])
            self.surface_data.GetPointData().SetScalars(colors_array)
            #super.__init__(mapper)
        self.actor.GetProperty().SetOpacity(1)

        self.add_vectors(vertices, faces)

    def add_vectors(self, vertices: np.ndarray, faces: np.ndarray):
        assert len(vertices.shape) == 2
        assert len(faces.shape) == 2
        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3

        # Add points
        [num_vertices, _] = vertices.shape
        for vertex_idx in range(num_vertices):
            self._vertices.InsertNextPoint(vertices[vertex_idx, 0], vertices[vertex_idx, 1], vertices[vertex_idx, 2])
        [num_faces, _] = faces.shape
        for face_idx in range(num_faces):
            self._faces.InsertNextCell(3)
            for corner_idx in range(3):
                self._faces.InsertCellPoint(faces[face_idx, corner_idx])
        # Allocate additional memory
        self._vertices.Resize(self.num_vertices + num_vertices)
        #self._faces.Resize(self.num_faces + num_faces*3)

        self._vertices.Modified()
        self._faces.Modified()


    def change_points(self, vertices: np.ndarray, faces: np.ndarray):
        assert len(vertices.shape) == 2
        assert len(faces.shape) == 2
        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3

        # Add points
        verticesPoints = vtk.vtkPoints()
        facesCells = vtk.vtkCellArray()
        [num_vertices, _] = vertices.shape
        for vertex_idx in range(num_vertices):
            verticesPoints.InsertNextPoint(vertices[vertex_idx, 0], vertices[vertex_idx, 1], vertices[vertex_idx, 2])
        [num_faces, _] = faces.shape
        for face_idx in range(num_faces):
            facesCells.InsertNextCell(3)
            for corner_idx in range(3):
                facesCells.InsertCellPoint(faces[face_idx, corner_idx])
        # Allocate additional memory
        verticesPoints.Resize(self.num_vertices + num_vertices)
        # self._faces.Resize(self.num_faces + num_faces*3)

        self._vertices.ShallowCopy(verticesPoints)
        self._faces.ShallowCopy(facesCells)
        self._vertices.Modified()
        self._faces.Modified()

    def updateTexture(self, p2pmap):
        VT1 = self.VT0[p2pmap]
        self.surface_data.GetPointData().SetTCoords(numpy_to_vtk(VT1))

class VTKPlane(VTKEntity3D):
    def __init__(self, center: np.ndarray, normal: np.ndarray):

        # VTK point cloud representation
        self._plane = vtk.vtkPlaneSource()
        self._plane.SetCenter(center)
        self._plane.SetNormal(normal)
        self._plane.Update()

        # VTK polygone(surface) representation
        self._faces = vtk.vtkCellArray()

        # Visualization Pipeline
        # - Data source
        plane_data = self._plane.GetOutput()
        # - Add the vector arrays as 3D Glyphs


        # - Map the data representation to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(plane_data)

        super().__init__(mapper)

class VTKLine(VTKEntity3D):
    def __init__(self, orig: np.ndarray, end: np.ndarray, color=None):

        # VTK point cloud representation
        self._line = vtk.vtkLineSource()
        self._line.SetPoint1(orig)
        self._line.SetPoint2(end)
        self._line.Update()

        # Visualization Pipeline
        # - Data source
        line_data = self._line.GetOutput()
        # - Add the vector arrays as 3D Glyphs


        # - Map the data representation to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(line_data)

        super().__init__(mapper)
        if color is None:
            self.actor.GetProperty().SetColor(1.0, 0., 0.)
        else:
            self.actor.GetProperty().SetColor(color[0], color[1], color[2])


class VTKTubeLine(VTKEntity3D):
    def __init__(self, orig: np.ndarray, end: np.ndarray, color=None):

        # VTK point cloud representation
        self._line = vtk.vtkLineSource()
        self._line.SetPoint1(orig)
        self._line.SetPoint2(end)
        self._line.Update()

        self.tubeFilter = vtk.vtkTubeFilter()
        self.tubeFilter.SetInputData(self._line.GetOutput())
        self.tubeFilter.SetRadius(.025) # default is .5
        self.tubeFilter.SetNumberOfSides(50)
        self.tubeFilter.Update()
        # Visualization Pipeline
        # - Data source
        tube_data = self.tubeFilter.GetOutput()
        # - Add the vector arrays as 3D Glyphs


        # - Map the data representation to graphics primitives
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(tube_data)

        super().__init__(mapper)
        if color is None:
            self.actor.GetProperty().SetColor(1.0, 0., 0.)
        else:
            self.actor.GetProperty().SetColor(color[0], color[1], color[2])


class VTKVisualization(object):
    def __init__(self):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.5, 0.5, 0.5)
        self.renderer.ResetCamera()

        # axes_actor = vtk.vtkAxesActor()
        # axes_actor.AxisLabelsOff()
        # self.renderer.AddActor(axes_actor)

        self.window = None

        self.camera = vtk.vtkCamera()
        self.camera.SetViewUp(0.0, 1.0, 0.0)
        self.camera.SetPosition(0.0, 0.0, +2.5)
        self.camera.SetFocalPoint(0.0, 0.0, 0.0)
        # self.camera.SetClippingRange(0.0, 100000)"""

        self.renderer.SetActiveCamera(self.camera)

    def add_entity(self, entity: VTKEntity3D):
        self.renderer.AddActor(entity.actor)

    def add_image(self, image):
        img_mapper = vtk.vtkImageMapper()
        img_actor = vtk.vtkActor2D()
        img_data = vtk.vtkImageData()
        img_data.SetDimensions(image.shape[0], image.shape[1], 1)
        img_data.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
        for x in range(0, image.shape[0]):
            for y in range(0, image.shape[1]):
                pixel = img_data.GetScalarPointer(x, y, 0)
                pixel = np.array(image[x, y, :])
        img_mapper.SetInputData(img_data)
        img_mapper.SetColorWindow(255)
        img_mapper.SetColorLevel(127.5)
        img_actor.SetMapper(img_mapper)
        self.renderer.AddActor(img_actor)

    def init(self, size=None):
        self.window = vtk.vtkRenderWindow()
        self.window.AddRenderer(self.renderer)
        self.window.Render()
        if size is None:
            self.window.SetSize(1200, 800)
        else:
            self.window.SetSize(size[0], size[1])
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.interactor.SetRenderWindow(self.window)

    def show(self):

        self.interactor.Start()

    def write(self, filename, background=None, size=None):
        self.window = vtk.vtkRenderWindow()

        self.window.SetOffScreenRendering(1)
        self.window.SetAlphaBitPlanes(1)
        if size is None:
            self.window.SetSize(1800, 1800)
        else:
            self.window.SetSize(size[0], size[1])
        windowToImageFilter = vtk.vtkWindowToImageFilter()
        if background is None:
            windowToImageFilter.SetInputBufferTypeToRGBA()
        else:
            self.renderer.SetBackground(background[0], background[1], background[2])
        self.window.AddRenderer(self.renderer)
        windowToImageFilter.SetInput(self.window)
        windowToImageFilter.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(windowToImageFilter.GetOutputPort())
        writer.Write()

    def change_camera(self, view, position, focal):
        self.camera = vtk.vtkCamera()
        self.camera.SetViewUp(view[0], view[1], view[2])
        self.camera.SetPosition(position[0], position[1], position[2])
        self.camera.SetFocalPoint(focal[0], focal[0], focal[0])
        self.camera.Pitch(0)
        self.camera.OrthogonalizeViewUp()
        # self.camera.SetClippingRange(0.0, 100000)

        self.renderer.SetActiveCamera(self.camera)

class VTKMultipleVizualization(object):
    # Not necessarily useful for everybody
    # A class to vizualize multiple object in the same window
    # Most ideas come from https://github.com/SRNFmatch/SRNFmatch_code/blob/master/src/input_output.py,
    # PlotGeodesic (and other vizualization functions)
    def __init__(self, N, sync=True):
        q, r = np.divmod(N, 4)

        if N <= 4:
            L_ind = np.arange(N)
            n_hwin = N
            n_vwin = 1
        elif N > 4 and N <= 8:
            L_ind = np.arange(N)
            n_hwin = 4
            n_vwin = 2
        else:
            L_ind = np.linspace(0, N - 1, 8)
            L_ind = np.rint(L_ind)
            L_ind = L_ind.astype(int)
            Nt = 8
            n_hwin = 4
            n_vwin = 2
        self.L_ind = L_ind
        self.window = vtk.vtkRenderWindow()

        self.camera = vtk.vtkCamera()
        self.camera.SetViewUp(0.0, 1.0, 0.0)
        self.camera.SetPosition(0.0, 0.0, +2.5)
        self.camera.SetFocalPoint(0.0, 0.0, 0.0)
        # self.camera.SetClippingRange(0.0, 10000)"""
        self.renderers = []
        for i in range(N):
            qi, ri = np.divmod(i, 4)
            ren = vtk.vtkRenderer()
            ren.SetBackground(0.5, 0.5, 0.5)
            self.window.AddRenderer(ren)
            ren.SetViewport(ri / n_hwin, 1 - (qi + 1) / n_vwin, (ri + 1) / n_hwin, 1 - qi / n_vwin)
            self.renderers.append(ren)
        if sync:
            for i in range(N - 1):
                self.renderers[i].SetActiveCamera(self.renderers[-1].GetActiveCamera())

    def show(self):
        self.window.Render()
        #self.window.SetSize(1200, 800)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.interactor.SetRenderWindow(self.window)
        self.interactor.Start()

    def add_entities(self, entities):
        for i, entity in enumerate(entities):
            self.renderers[i].AddActor(entity.actor)

    def add_entity(self, entity, i):
        self.renderers[i].AddActor(entity.actor)