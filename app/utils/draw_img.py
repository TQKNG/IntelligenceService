def draw_img(data):
    try:
        img = data.draw_mermaid_png()

        # Save image to file
        with open('graph_image.png','wb') as f:
            f.write(img)

    except Exception as e:
        print(e)