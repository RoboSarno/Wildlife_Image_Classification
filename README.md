# Wildlife Image Classification
**Project by: Franco Sanchez, Jerry Montes, Luisa Gonzalez, Robert Sarno**

## Problem Statement

Our team is tasked with aiding conservationist efforts to study the population and distribution of seven animal groups at Tai National Park by developing an image classification model that can identify each species captured by camera traps throughout the park. **Our success criteria is to develop a model that outperforms a baseline model that yields accuracies based on proportion of classes in the model.**

### Background

Located in Côte d’Ivoire, Taï National Park consists of 330,000 hectares of the last remaining West African tropical forest.  Recognized as a UNESCO Natural World Heritage Site since 1982, the park continues to be exploited for timber, with deforestation increasing dramatically within the park’s boundaries over the last few decades.  Several animal species that call the park home have been the target of poachers for centuries, which has greatly affected species population sizes. Profitable to few and detrimental to many, illegal gold mining has resulted in the destruction of large swaths of the park.  Even more acres have been cleared through slash and burn agriculture by local farmers who have converted protected land into illegal cocoa and coffee bean farms.  These disturbances have affected not only the park’s flora, but also its fauna, which includes several endangered species.  

In order to quantify the impact human actions have had on the park’s residents, conservationists at the Wild Chimpanzee Foundation and the Max Planck Institute for Evolutionary Anthropology have installed camera traps that are triggered by heat or motion throughout the park in an effort to track seven types of animal species for conservation efforts. Unless the images are classified, researchers would have to manually search through the tens of thousands of pictures taken in order to find those that contain the particular species they are interested in studying.  


## About each animal class:

The seven species groups the conservationists are interested in studying are: antelope/duiker, bird, civet/genet, hog, leopard, monkey/prosimian, and rodent. Some images taken by the camera traps did not contain an animal, and our team sought to ensure the model would classify these as blank.  


**Antelope/duiker:**
The park is home to several antelope and duiker species, both of which belong to the Bovidae family.  Most of these species are diurnal, with the bay duiker being a notable exception.  As such, we expected the majority of the images that captured this species group to have been taken during the day. These animals are often poached for their meat, coat, and use in traditional medicine.  Several species are threatened by extinction due to deforestation and the resulting habitat loss.

**Birds:**
The bird population of Taï National Park consists of over 250 different species, and this population is not currently endangered.

**Civet/Genet:**
Civets and genets both belong to the Viverridae family and are mostly nocturnal.  Their coat contains spots and stripes, and they are captured and kept by those who want an exotic pet and those who seek musk, which is used as an ingredient in exotic perfumes and traditional medicine.  These animals are also hunted for their meat and fur.

**Hogs:**
Hogs are generally nocturnal, but in cold periods they are commonly seen during daylight hours. They are threatened by deforestation and by hunting for food. The species, however, is currently classified as ‘Least Concern’.

**Leopards:**
A single leopard species calls the park home, the panthera pardus pardus. TLike civets and genets, leopards possess a spotted coat.  This animal is mostly nocturnal and is hunted for its fur, claws, whiskers, tails, and meat.  As the species it preys on experience a population decline, the leopard will have fewer feeding sources.  Like the aforementioned species, this species is threatened by habitat fragmentation.  

**Monkeys/Promisians:**
The park is home to eleven primate species. Three species are nocturnal and are not under threat. The rest are diurnal and a few of those species are threatened by over-hunting and deforestation. Prosimians are primarily nocturnal and their population is threatened by habitat loss due to mining, agriculture, and hunting.

**Rodents:**
Tai National Park has several rodent species, including rusty-bellied brush-furred rat, edward’s swamp rat, and the woodland dormouse. Rodents are both nocturnal and diurnal.

______

### Approach

We will implement a convolutional neural network build upon TensorFlow's API to classify images from the trap cameras.  

#### Notebook/Code

|Index|Title|Description|
|---|---|---|
|1|[Early Data Exploration]() | This notebook captures all the EDA that explores patterns in the animal images.|
|2|[Modeling]() | This notebook captures all the image preprocessing and neural network model development.|

#### EDA Process

In the EDA process, we explored the total number of observations of each animal class that is present in the training data. We also discovered that there are a total of 148 sites with camera traps.

We further explored which sites have the highest observations of animals and noticed that few sites hold a high concentration of animal sightings. Images from 20 out of 148 sites, obtain about half of all observations. Our EDA also highlighted that each animal class has a preference for specific sites in the park. This can be helpful to note for conservationist as they plan how to best care for each species.  

The data provided to our team was split into train and test sets, while the training data had labels, our test data did not. In order to test our model, we created a validation dataset from the training data. By exploring the distribution of observations we were able to create a validation set that represents about 10% of the training data and hold observations of each animal type without over-representing any class. We also decided to separate the train and validation sets by sites, so that the model can test on sites it had not yet observed before and thereby preventing the model from generalizing based on site instead of by animal. This however, presented a few challenges for our modeling because we found it difficult to create a validation set that mirrored the proportion of classes in the training class (we were unable to code a loop to do this).

**Limitations highlighted by the EDA Process**

During our EDA we also explored the training images to understand the challenges they present. Most of the images, for instance are not focused, while in some images details are washed out due to brightness or darkness. There are also images where the presence of an animal is very subtle such that the image appears to be blank. Yet, in other images, the animals are really close to the camera lens and it is difficult to discern to which class it belongs. We also noticed that leopards and civets share very similar coats.

We also noticed that several of the images displayed a chyron in the lower 7%. This can present a challenge where the model may generalize based on the chyron instead of animal species.


#### Modeling/Metrics Process

We initially started with a baseline model (simple architecture).  This model’s accuracy score was extremely high but validation accuracy was extremely low.  We noticed that we needed to add more complexity and data augmentation.

**Data Augmentation:**
- Due to watermarks inconsistently on images we calculated that the watermarks covered 7% of the height of the image.  We incorporated a height shift to account for this.

- Due to the photographs being taken during different times of the day some of the classification could be skewed.  We decided to add a preprocessing function augmentation to reduce the color of the image by making it to grayscale instead of RGB.

- We also added a horizontal flip to account for the possible locations that birds, monkey, and rodents could be located in the images.

**Complexity:**
- After further review of the low validation accuracy we noticed that the model was extremely overfit.  This was caused by the imbalance of the classification distribution in the train and validation set.  To account for this we used Image Data Generator and specified a validation split.

- After several models focusing on reducing overfitting, and adding complexity we came to the Final Model Architecture of:
  - Sequential
  - Input Layer:
    1. EfficientNetB0
    2. PReLU
    3. GlobalAveragePooling2D
    4. Flatten
    5. Dropout
  - Hidden Layer 1:
    1. Dense
    2. Dropout
  - Hidden Layer 2:
    1. Dense
    2. Dropout
  - Output Layer :
    1. Dense

**Metrics:**
- We used a loss of categorical cross entropy and Adam optimizer and focused on the following metrics: accuracy, recall, and precision.

- Our Initial Model’s performance -
Accuracy: 0.62, Validation Accuracy: 0.46, Recall: 0.28, Validation Recall: 0.28, Precision: 0.86, Validation Precision: 0.59.

- Our Final Model’s performance -
Accuracy: 0.62, Validation Accuracy: 0.46, Recall: 0.28, Validation Recall: 0.28, Precision: 0.86, Validation Precision: 0.59.

**Play with our mode model on our steamlit app here:**
(add streamlit link here)

-----------
### Recommendations:

Given that some species showed a clear preference for specific sites, we would like to incorporate a Multi-data type Neural Network to predict the Site location and the image data to classify animals in each image.  Additionally we would like to augment the images even further by introducing vertical and horizontal flips in the images that would enable the model to detect an animal whether it is standing up or hanging upside down while also increasing the number of images in the training and validation sets.  

We are also curious to see what the effect of converting all of the images to grayscale would be, as we expect this would highlight each image’s distinct features and textures.  Finally, we would be curious to see what the model’s performance would be like if we removed the site information altogether and trained the model using only the images


### Conclusions:

Counting the number of images taken by the camera traps that contained each animal species we are currently studying allows us to estimate the current population size of each species group in the park.  By understanding these quantities, the conservationists will be better able to direct resources where they are most needed.  Noticing a precipitous decline in the number of images that contain leopards, for instance, would indicate that more resources should be directed towards saving these big cats from extinction.  

While it might be unfeasible to patrol the entire park given its enormous size, becoming familiar with each species’ favorite sites would allow for an informed decision about which areas to monitor in an attempt to prevent poaching.  The species most coveted by poachers could benefit from having increased patrolling in and around their favorite sites in the park.

Our final model was better able to predict the species group of the animal that triggered the camera trap to take a picture than simply looking at the normalized distribution of classes in the training data using a baseline model.  That is, the model’s predictions were more accurate than simply guessing the species contained in each image based solely on the percentage of images in the training data that were labeled as having captured each species group.  As such, our model has outperformed the baseline model, and we have successfully solved our data science problem.

Once classified, the images will provide a wealth of information about all of the species that pass by the camera traps.  Scientists were recently able to identify the first known cases of leprosy in wild chimpanzees by analyzing images of chimpanzees taken by camera traps.  These traps also allow scientists to continue learning more about animal behavior in the wild, such as the accumulative stone throwing behavior first noticed in camera trap footage of West African chimpanzees.

__________


### Sources
- https://whc.unesco.org/en/list/195
- https://whc.unesco.org/en/list/195/documents/
- https://www.worldheritagesite.org/list/Taï+National+Park
- https://www.researchgate.net/publication/-227901726_Spatial_organization_of_leopards_Panthera_pardus_in_Tai_National_Park_Ivory_Coast_Is_rainforest_habitat_a_'tropical_haven'
- http://world-heritage-datasheets.unep-wcmc.org/datasheet/output/site/tai-national-park/
- https://www.britannica.com/animal/duiker
- https://animaldiversity.org/accounts/Panthera_pardus/
- https://en.wikipedia.org/wiki/Giant_forest_hog
- https://hubble-live-assets.s3.amazonaws.com/psgb/-redactor2_assets/files/152/Ryan_Covey_CWPFinal_small.pdf
- https://www.iucn.org/content/quarter-antelope-species-danger-extinction
- https://www.britannica.com/animal/civet-mammal-Viverridae-family
- https://animals.net/genet/
- https://www.awf.org/wildlife-conservation/leopard
- https://www.nature.com/articles/s41586-021-03968-4
- https://animalia.bio/african-civet
- https://www.cocoalife.org/in-the-cocoa-origins/cocoa-life-in-cote-divoire


### Images
- Cote D’Ivoire map: https://cdn.britannica.com/92/5092-050-1B901E58/Cote-dIvoire-map-features-locator.jpg
- Camera trap: https://savingnature.com/wp-content/uploads/2021/04/shutterstock_1750405736-1024x630.jpg
- Timber Logging: https://cdn.powerofpositivity.com/wp-content/uploads/2020/10/Logging-Industry-May-Turn-Amazon-Rainforests-into-Savannahs.jpg
- Poaching: https://images.hindustantimes.com/rf/image_size_960x540/HT/p2/2018/01/31/Pictures/leopard-rescue_034d36e4-06a9-11e8-987c-1603f9800600.jpg
- Gold mining: http://afrique.le360.ma/sites/default/files/assets/images/2016/09/une-equipe-d-orpailleurs.jpg
- Farming: https://thumbs-prod.si-cdn.com/nMercL7OElV1pCNbG9ZLCwdi3d8=/fit-in/1072x0/https://public-media.si-cdn.com/filer/3c/63/3c630860-1c56-472a-b790-1d94cc79c475/dsc_2435.jpg
- Bay_duiker: https://i.pinimg.com/originals/36/68/7a/36687a3d729153fdeb5234d0abbb9006.jpg
- Black throated coucal: ​​https://cdn.download.ams.birds.cornell.edu/api/v1/asset/249916851/1800
- African civet: https://content.eol.org/data/media/84/f1/40/7.CalPhotos_0000_0000_0113_0858.jpg
- Giant forest hog: https://vignette.wikia.nocookie.net/cryptidarchives/images/7/75/Giant_forest_hog.png/revision/latest?cb=20180921201715
- Leopard: https://i.ytimg.com/vi/lTRHxmvraHU/maxresdefault.jpg
- Soorty Mangabey: https://news.berkeley.edu/wp-content/uploads/2019/08/sootymangabey750px.jpg
- Woodland dormouse: https://www.worldlifeexpectancy.com/images/a/w/b/muscardinus-avellanarius/muscardinus-avellanarius.jpg
- Chimpanzee with leprosy: https://interestingengineering.com/leprosy-found-in-wild-chimps-for-first-time-ever