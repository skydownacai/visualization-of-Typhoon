# _*_ coding:utf-8 _*_
import imageio

def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        print(image_name)
        frames.append(imageio.imread(image_name))
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return

def main():
    image_list = ['预警线frame/{}.png'.format(60-i) for i in range(61)]
    for i in range(10):
        image_list .append('预警线frame/{}.png'.format(0))
    gif_name = '预警线变动.gif'
    duration = 0.2
    create_gif(image_list, gif_name, duration)
main()