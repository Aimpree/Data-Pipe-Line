import cv2
import os
import time

def RGB2BINARY_Transform(Input: str, Output_Folder: str, Save_As_Name: str):

    os.makedirs(Output_Folder, exist_ok=True)
    
    Gray_Image = cv2.imread(Input, cv2.IMREAD_GRAYSCALE)
    
    _, Tran_Image = cv2.threshold(Gray_Image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    Output_Path = os.path.join(Output_Folder, Save_As_Name)

    cv2.imwrite(Output_Path, Tran_Image)


def Label_Extract(Input: str, Output_Folder: str, Save_As_Name: str, 
                  Class_Name: str, Image_W: int = 5100, Image_H: int = 3750):
    
    os.makedirs(Output_Folder, exist_ok=True)

    Image = cv2.imread(Input, cv2.IMREAD_GRAYSCALE)

    Contours, hierarchy = cv2.findContours(image=Image, 
                                           mode=cv2.RETR_TREE, 
                                           method=cv2.CHAIN_APPROX_SIMPLE)

    Image_copy = cv2.cvtColor(Image, cv2.COLOR_GRAY2BGR)

    Min_Contour_Area = 5000  

    Label_list = []

    for Contour in Contours:

        W = Image_W
        H = Image_H
        
        Contour_area = cv2.contourArea(Contour)
        
        
        if Contour_area > Min_Contour_Area:
            
            x, y, w, h = cv2.boundingRect(Contour)
    
            ## YOLOv8 PyTorch TXT

            center_x = x + (w / 2)
            center_y = y + (h / 2)

            Nor_center_x = center_x / W
            Nor_center_y = center_y / H

            Nor_W = w/W
            Nor_h = h/H

            Label_list.append(Class_Name)

            Label_list.append(Nor_center_x)
            Label_list.append(Nor_center_y)

            Label_list.append(Nor_W)
            Label_list.append(Nor_h)

            cv2.rectangle(Image_copy, (x,y), (x + w, y +h), (0, 255, 0), 2)

    Output_Path = os.path.join(Output_Folder, Save_As_Name)

    cv2.imwrite(Output_Path, Image_copy)

    return Label_list


def Create_TXT(Input_List_Of_Data: list, Number_Of_Chunk: 
               int, Output_Name: str, Output_Folder: str):

    os.makedirs(Output_Folder, exist_ok=True)
   
    def chunk_list(lst, chunk_size):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    TXT_File_Path = f'{Output_Folder}/{Output_Name}.txt'

    chunked_data = chunk_list(Input_List_Of_Data, Number_Of_Chunk)

    with open(TXT_File_Path, 'w') as file:
        for row in chunked_data:
            file.write(' '.join(map(str, row)) + '\n')


def main(Input: str, Class_Number: int, Number_of_Chunk: int):

    Final_Output = 'Process Images'
    os.makedirs(Final_Output, exist_ok=True)

    Count_Origin_Images = os.listdir(Input)
    Output_Binary_Images = f'{Final_Output}/Binary Images'
    Output_Label_File = f'{Final_Output}/Label Text File'

    for i in range(len(Count_Origin_Images)):
        
        RGB2BINARY_Transform(f'{Input}/{Count_Origin_Images[i]}', 
                             Output_Binary_Images, 
                             f'BR_{Count_Origin_Images[i]}')
    
    time.sleep(2)

    Count_Binary_Images = os.listdir(Output_Binary_Images)

    for i in range(len(Count_Binary_Images)):

        List_of_Data = Label_Extract(f'{Output_Binary_Images}/{Count_Binary_Images[i]}', 
                                     f'{Final_Output}/Display Boundary Box Images', 
                                     f'DP_{Count_Binary_Images[i]}', Class_Number)
        
        Create_TXT(List_of_Data, Number_of_Chunk, 
                   Count_Origin_Images[i], Output_Label_File)

    
    print("Compleat Created")
    

if __name__ == "__main__":

    Input_Path = 'C:/Users/Lenovo/Desktop/Rice dataset2/Test'
    Number_of_Chunk = 5
    Class_Number = 0

    main(Input_Path, Class_Number, Number_of_Chunk)
