"""Bounded local jaw alignment for side references with an open mouth."""
import numpy as np


def mouth_landmark(mask, box):
    from scipy.ndimage import gaussian_filter1d
    y0,y1,x0,x1=box
    # Side-view heads are in the upper third. Find a clear inward step at
    # the front outline, excluding crown spikes and the neck base.
    top=max(y0+int((y1-y0)*.12),0)
    bottom=min(y0+int((y1-y0)*.30),len(mask)-2)
    head=mask[y0:bottom]
    yy,xx=np.nonzero(head)
    if not len(xx):
        return None
    right=float(np.mean(xx))>(x0+x1)/2
    profile=np.zeros(len(mask),float)
    for y in range(y0,len(mask)):
        positions=np.flatnonzero(mask[y])
        if len(positions):
            profile[y]=positions.max() if right else -positions.min()
        elif y:
            profile[y]=profile[y-1]
    derivative=np.gradient(gaussian_filter1d(profile,1))
    candidates=np.flatnonzero(derivative[top:bottom]<-4)
    if not len(candidates):
        return None
    start=top+int(candidates[0])
    stop=start+1
    while stop<bottom and derivative[stop]<-4:
        stop+=1
    notch=start+int(np.argmin(derivative[start:stop]))
    strength=float(-derivative[notch])
    return notch,strength,right


def align_side_mouth(mesh,source):
    from PIL import Image
    from backend.aligned_vertex import camera
    from backend.reference_texture import depth_buffer
    rgba=np.asarray(source.convert('RGBA').resize((512,512),Image.Resampling.LANCZOS))
    mask=rgba[:,:,3]>128
    yy,xx=np.nonzero(mask)
    if not len(xx):
        return {'applied':False,'reason':'empty_reference'}
    box=(yy.min(),yy.max(),xx.min(),xx.max())
    vertices=np.asarray(mesh.vertices).copy()
    projection,_=camera(vertices,0,0,(box[2],box[0],box[3],box[1]))
    silhouette=depth_buffer(projection,np.asarray(mesh.faces),512,512)>-1e9
    target=mouth_landmark(mask,box)
    current=mouth_landmark(silhouette,box)
    if target is None or current is None or min(target[1],current[1])<4 or target[2]!=current[2]:
        return {'applied':False,'reason':'no_clear_mouth_notch'}
    shift=float(target[0]-current[0])
    height=box[1]-box[0]
    if abs(shift)<2 or abs(shift)>height*.08:
        return {'applied':False,'reason':'shift_outside_safe_range','shift_pixels':shift}
    # Smoothly move the mouth region; leave crown, neck, body and depth intact.
    py=projection[:,1]
    smooth=lambda t: np.clip(t,0,1)**2*(3-2*np.clip(t,0,1))
    rise=smooth((py-(current[0]-height*.16))/(height*.10))
    fall=1-smooth((py-(current[0]+height*.04))/(height*.14))
    hy,hx=np.nonzero(mask[box[0]:box[0]+int(height*.32)])
    hmin,hmax=float(hx.min()),float(hx.max())
    px=projection[:,0]
    front=smooth((px-hmin)/(max(hmax-hmin,1)*.4)) if target[2] else smooth((hmax-px)/(max(hmax-hmin,1)*.4))
    weight=rise*fall*front
    vertices[:,1]-=shift*weight*np.ptp(vertices[:,1])/height
    mesh.vertices=vertices
    return {'applied':True,'source_notch':int(target[0]),'mesh_notch':int(current[0]),
            'shift_pixels':shift,'max_displacement':float(np.max(np.abs(shift*weight*np.ptp(vertices[:,1])/height)))}

