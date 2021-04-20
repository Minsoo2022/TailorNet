import os
import json
import argparse
from psbody.mesh import Mesh
from utils.interpenetration import remove_interpenetration_fast, remove_interpenetration_fast_custom

# Set output path where inference results will be stored

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataroot", default="../data/Merging_Splitting/1_0/")
    parser.add_argument("--gpu_ids", type=str, default="0")
    opt = parser.parse_args()
    return opt

def custom_obj_save(file_dir, file_name, base_name, object):
    file_dir = os.path.join(file_dir, file_name)
    with open('./obj_info.txt', 'r') as f_info:
        info = f_info.readlines()
    with open('./mtl_info.txt', 'r') as m_info:
        mtl_info = m_info.readlines()
    with open(file_dir.replace('.obj', '.mtl'), 'w') as fm:
        fm.write(''.join(mtl_info))
        fm.write('map_Kd %s_octopus.jpg' % (base_name))
    with open(file_dir, 'w') as fp:
        for v in object.v:
            fp.write('v {:f} {:f} {:f}\n'.format(v[0], v[1], v[2]))
        fp.write(''.join(info))
        fp.write('\nusemtl Mymtl')

def custom_obj_save_with_mtl(file_dir, base_name, file_name, vertices, faces, vts, vns):
    with open('./obj_info.txt', 'r') as f_info:
        info = f_info.readlines()
    with open('./mtl_info.txt', 'r') as m_info:
        mtl_info = m_info.readlines()
    vt_info = [i for i in info if i.startswith('vt')]
    file_dir = os.path.join(file_dir, file_name)
    with open(file_dir, 'w') as fp:
        fp.write('mtlib %s\n'%file_name.replace('.obj','.mtl'))
        for v in vertices:
            fp.write('v {:f} {:f} {:f}\n'.format(v[0], v[1], v[2]))
        for f, vt, vn in zip(faces, vts, vns):
            fp.write('f {:d}/{:d}/{:d} {:d}/{:d}/{:d} {:d}/{:d}/{:d}\n'.format(f[0], vt[0], vn[0], f[1], vt[1], vn[1], f[2], vt[2], vn[2],))
        fp.write(''.join(vt_info))
        fp.write('\nusemtl Mymtl')
    with open(file_dir.replace('.obj', '.mtl'), 'w') as fm:
        fm.write(''.join(mtl_info))
        fm.write('map_Kd %s_octopus.jpg' % (base_name))


def custom_obj_save1(file_dir, file_name, vertices, faces, color=None):
    file_dir = os.path.join(file_dir, file_name)
    with open(file_dir, 'w') as fp:
        if color is None:
            for v in vertices:
                fp.write('v {:f} {:f} {:f}\n'.format(v[0], v[1], v[2]))
        else:
            for v, c in zip(vertices, color):
                fp.write('v {:f} {:f} {:f} {:f} {:f} {:f}\n'.format(v[0], v[1], v[2], c[0], c[1], c[2]))
        for f in faces:
            fp.write('f {:f} {:f} {:f}\n'.format(f[0], f[1], f[2]))


def convert(o):
    return int(o)

class ObjLoader(object):
    def __init__(self, fileName):
        self.vertices = []
        self.faces_vt_vn = []
        self.v_to_vt = {}
        ##
        try:
            f = open(fileName)
            for line in f:
                if line[:2] == "v ":
                    index1 = line.find(" ") + 1
                    index2 = line.find(" ", index1 + 1)
                    index3 = line.find(" ", index2 + 1)

                    vertex = (float(line[index1:index2]), float(line[index2:index3]), float(line[index3:-1]))
                    vertex = (round(vertex[0], 4), round(vertex[1], 4), round(vertex[2], 4))
                    self.vertices.append(vertex)

                elif line[0] == "f":
                    string = line.replace("//", "/")
                    ##
                    i = string.find(" ") + 1
                    face = []
                    for item in range(string.count(" ")):
                        if string.find(" ", i) == -1:
                            face.append(string[i:-1])
                            break
                        face.append(string[i:string.find(" ", i)])
                        i = string.find(" ", i) + 1
                    ##
                    self.faces_vt_vn.append(tuple(face))

            f.close()

            self.v_to_vt = {int(x.split('/')[0]): int(x.split('/')[1]) for face in self.faces_vt_vn for x in face}
            self.faces = [(int(a.split('/')[0]), int(b.split('/')[0]), int(c.split('/')[0])) for a, b, c in
                          self.faces_vt_vn]
            self.vts = [(int(a.split('/')[1]), int(b.split('/')[1]), int(c.split('/')[1])) for a, b, c in
                        self.faces_vt_vn]
            self.vns = [(int(a.split('/')[2]), int(b.split('/')[2]), int(c.split('/')[2])) for a, b, c in
                        self.faces_vt_vn]

        except IOError:
            print(".obj file not found.")

def make_body_mesh_with_clothes(opt):
    #file_dir = '../data/BCNet_merging_mesh/4_0/'
    file_dir = opt.dataroot
    file_name = [i for i in file_dir.split('/') if len(i)>0][-1]
    body = Mesh(filename=os.path.join(file_dir,f'{file_name}_ori.obj'))
    up = Mesh(filename=os.path.join(file_dir,f'{file_name}_up.obj'))
    bottom = Mesh(filename=os.path.join(file_dir,f'{file_name}_bottom.obj'))

    up = remove_interpenetration_fast(up, body)
    #pred_body, indices_up = remove_interpenetration_fast_custom(body, up, threshold=0.0010, direction_threshold=0.0003)
    pred_body, indices_up = remove_interpenetration_fast_custom(body, up, thresh_dir=0.2, thresh_dist=0.001)
    #pred_body.write_obj(os.path.join(file_dir, 'deformed_body_with_shirt.obj'))

    bottom = remove_interpenetration_fast(bottom, pred_body)
    pred_body, indices_down = remove_interpenetration_fast_custom(pred_body, bottom, thresh_dir=0.2, thresh_dist=0.001)
    #pred_body, indices_down = remove_interpenetration_fast_custom(pred_body, bottom, threshold=0.0005, direction_threshold=0.0003)
    #pred_body.write_obj(os.path.join(file_dir, 'deformed_body_with_pant_shirt.obj'))
    #pred_body.write_obj(os.path.join(file_dir, f'{file_name}_with_garment.obj'))
    custom_obj_save(file_dir, f'{file_name}_with_garment.obj', file_name, pred_body)
    indices_info = {'indices_down' : [int(i) for i in list(indices_down)],
                    'indices_up' : [int(i) for i in list(indices_up)]}
    with open(os.path.join(file_dir, 'indices_info.json'), 'w') as json_file:
        json.dump(indices_info, json_file)


def split_clothes_mesh(opt):
    with open('./obj_info.txt', 'r') as f_info:
        obj_info = f_info.readlines()
    vt_info = [(float(a), float(b)) for a, b in [i[3:16].split(' ') for i in obj_info if i.startswith('vt')]]

    # %%
    file_dir = opt.dataroot
    base_name  = [i for i in file_dir.split('/') if len(i) > 0][-1]
    folder_dir = opt.dataroot
    with open(os.path.join(folder_dir, 'indices_info.json'), "r") as f:
        indices_info = json.load(f)
    indices_info['indices_up'] = [i + 1 for i in indices_info['indices_up']]
    indices_info['indices_down'] = [i + 1 for i in indices_info['indices_down']]
    obj_info = ObjLoader(os.path.join(folder_dir, f'{base_name}_with_garment.obj'))

    # %%

    up_vertices_list = [obj_info.vertices[index - 1] for index in indices_info['indices_up']]

    up_vertices_long_list = []
    up_newv_to_oldv = {}
    num = 1
    for i in range(8000):
        if i in indices_info['indices_up']:
            up_vertices_long_list.append(num)
            up_newv_to_oldv[num] = i
            num += 1
        else:
            up_vertices_long_list.append(-1)

    up_faces_list = []
    up_vts_list = []
    up_vns_list = []
    for face, vt, vn in zip(obj_info.faces, obj_info.vts, obj_info.vns):
        if up_vertices_long_list[face[0]] == -1 or up_vertices_long_list[face[1]] == -1 or up_vertices_long_list[face[2]] == -1:
            continue
        up_faces_list.append(
            (up_vertices_long_list[face[0]], up_vertices_long_list[face[1]], up_vertices_long_list[face[2]]))
        up_vts_list.append(vt)
        up_vns_list.append(vn)


    down_vertices_list = [obj_info.vertices[index - 1] for index in indices_info['indices_down']]

    down_vertices_long_list = []
    down_newv_to_oldv = {}
    num = 1
    for i in range(8000):
        if i in indices_info['indices_down']:
            down_vertices_long_list.append(num)
            down_newv_to_oldv[num] = i
            num += 1
        else:
            down_vertices_long_list.append(-1)

    down_faces_list = []
    down_vts_list = []
    down_vns_list = []
    for face, vt, vn in zip(obj_info.faces, obj_info.vts, obj_info.vns):
        if down_vertices_long_list[face[0]] == -1 or down_vertices_long_list[face[1]] == -1 or down_vertices_long_list[
            face[2]] == -1:
            continue
        down_faces_list.append(
            (down_vertices_long_list[face[0]], down_vertices_long_list[face[1]], down_vertices_long_list[face[2]]))
        down_vts_list.append(vt)
        down_vns_list.append(vn)

    # %%
    custom_obj_save_with_mtl(folder_dir, base_name, f'{base_name}_splited_up.obj', up_vertices_list, up_faces_list,
                             up_vts_list, up_vns_list)
    custom_obj_save_with_mtl(folder_dir, base_name, f'{base_name}_splited_down.obj', down_vertices_list,
                             down_faces_list, down_vts_list, down_vns_list)


if __name__ == '__main__':
    opt= get_opt()
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    make_body_mesh_with_clothes(opt)
    split_clothes_mesh(opt)
