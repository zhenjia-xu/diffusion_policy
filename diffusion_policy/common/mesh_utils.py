def meshwrite(filename, verts, colors, faces=None):
    """Save 3D mesh to a polygon .ply file.
    Args:
        filename: string; path to mesh file. (suffix should be .ply)
        verts: [N, 3]. Coordinates of each vertex
        colors: [N, 3]. RGB or each vertex. (type: uint8)
        faces: (optional) [M, 4]
    """
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    if faces is not None:
        ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write(
            "%f %f %f %d %d %d\n" %
            (verts[i, 0], verts[i, 1], verts[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))

    # Write face list
    if faces is not None:
        for i in range(faces.shape[0]):
            ply_file.write("4 %d %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2], faces[i, 3]))

    ply_file.close()