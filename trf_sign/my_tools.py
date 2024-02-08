
import numpy as np
import matplotlib.pyplot as plt
import pwk as pwk



def show_prediction( prediction, x, y, x_meta ):

    # ---- A prediction is just the output layer
    pwk.subtitle("Output layer from model is (x100) :")
    with np.printoptions(precision=2, suppress=True, linewidth=95):
        print(prediction*100)

    # ---- Graphic visualisation
    #
    pwk.subtitle("Graphically :")
    plt.figure(figsize=(8,2))
    plt.bar(range(43), prediction[0], align='center', alpha=0.5)
    plt.ylabel('Probability')
    plt.ylim((0,1))
    plt.xlabel('Class')
    plt.title('Trafic Sign prediction')
    pwk.save_fig('05-prediction-proba')
    plt.show()

    # ---- Predict class
    #
    p = np.argmax(prediction)

    # ---- Show result
    #
    pwk.subtitle('In pictures :')
    print("\nThe image :               Prediction :            Real stuff:")
    pwk.plot_images([x,x_meta[p], x_meta[y]], [p,p,y], range(3),  columns=3,  x_size=1.5, y_size=1.5, save_as='06-prediction-images')

    if p==y:
        print("YEEES ! that's right!")
    else:
        print("oups, that's wrong ;-(")