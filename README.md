## Problem Statement: Mushroom Edibility Classification Using Machine Learning

When exploring wilderness areas, trekking, or camping in remote locations, individuals may encounter wild mushrooms and other naturally occurring food sources. Mushrooms are known to be rich in nutrients and can serve as a valuable food source. However, distinguishing edible mushrooms from poisonous or toxic varieties is challenging without proper knowledge and expertise.

Consuming a poisonous mushroom can lead to severe health complications and, in some cases, life-threatening consequences. These risks become particularly concerning in remote areas where immediate medical assistance may not be accessible.

This project aims to develop a **Machine Learning-based Mushroom Classification Model** that analyzes the characteristics of a mushroom and predicts whether it is classified as edible or poisonous. By leveraging machine learning techniques and mushroom classification datasets, the model seeks to provide an additional layer of awareness and risk prevention when encountering unfamiliar mushroom species.

The ultimate objective is to explore how machine learning can support informed decision-making, reduce the risk of accidental mushroom poisoning, and improve safety during outdoor activities and wilderness exploration.

# MushroomClassifier
### Dataset Selection and Description

For this project, we selected the Mushroom Classification Dataset from the UCI Machine Learning Repository. The dataset consists of 8,124 mushroom observations, each described by 22 categorical input features and one target variable.

The input features represent various physical and biological characteristics of mushrooms, including cap shape, cap color, odor, gill characteristics, stalk structure, ring type, spore print color, population, and habitat.

The target variable, `class`, contains two possible classifications: edible (`e`) and poisonous (`p`). This makes the dataset suitable for a supervised binary classification problem, where the objective is to learn the relationship between mushroom characteristics and their corresponding edibility labels.

The dataset contains 4,208 edible mushroom records and 3,916 poisonous mushroom records, providing a relatively balanced distribution between the two classes. All input features are categorical, making data preprocessing and categorical encoding important steps before applying machine learning algorithms.

I selected this dataset because it provides an opportunity to explore different classification techniques and evaluate their ability to distinguish between two classes. The project also demonstrates how machine learning can be applied to a practical problem involving food safety and risk awareness.

uploaded dataset must have the data in the below mentioned format with no headers, use the data uploaded by David W. Aha for better results, as that dataset is used for training.

    1. cap-shape:                bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
    2. cap-surface:              fibrous=f,grooves=g,scaly=y,smooth=s
    3. cap-color:                brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
    4. bruises?:                 bruises=t,no=f
    5. odor:                     almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
    6. gill-attachment:          attached=a,descending=d,free=f,notched=n
    7. gill-spacing:             close=c,crowded=w,distant=d
    8. gill-size:                broad=b,narrow=n
    9. gill-color:               black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
    10. stalk-shape:              enlarging=e,tapering=t
    11. stalk-root:               bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
    12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
    13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
    14. stalk-color-above-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
    15. stalk-color-below-ring:   brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
    16. veil-type:                partial=p,universal=u
    17. veil-color:               brown=n,orange=o,white=w,yellow=y
    18. ring-number:              none=n,one=o,two=t
    19. ring-type:                cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
    20. spore-print-color:        black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
    21. population:               abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
    22. habitat:                  grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d

