
#python test/test.py

import disasteragent

if __name__ == "__main__":

    # result = disasteragent.run(
    #     pre_image_path="/local_data/lizhijian/dataset/DisasterM3_Bench/test_images/joplin_tornado_00000041_pre_disaster.png",
    #     post_image_path="/local_data/lizhijian/dataset/DisasterM3_Bench/test_images/joplin_tornado_00000041_post_disaster.png",
    #     task_description="Analyze both the pre-disaster and post-disaster images to answer the following question. Choose the best option from the candidate options provided. Question: What is the total number of completely destroyed buildings? Options: A:3, B:7, C:14, D:17, E:5. Your task is to respond with ONLY the capital letter of the correct option (e.g., C). Do not include any explanation or other text.",
    # )

    result = disasteragent.run(
         pre_image_path="/local_data/lizhijian/dataset/DisasterM3_Bench/test_images/nepal_flooding_00000548_pre_disaster.png",
         post_image_path="/local_data/lizhijian/dataset/DisasterM3_Bench/test_images/nepal_flooding_00000548_post_disaster.png",
         task_description="Generate a comprehensive disaster report, including the description of the disaster and recovery suggestions.",
    )    