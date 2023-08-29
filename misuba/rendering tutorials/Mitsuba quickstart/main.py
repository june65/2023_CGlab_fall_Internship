import mitsuba as mi

mi.set_variant('scalar_rgb')

scene = mi.load_file("./scenes/cbox.xml")

image = mi.render(scene, spp=256)


mi.util.write_bitmap("my_first_render.png", image)
mi.util.write_bitmap("my_first_render.exr", image)


# img = mi.render(mi.load_dict(mi.cornell_box()))

# mi.Bitmap(img).write('cbox.exr')