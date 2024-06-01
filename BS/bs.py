



if __name__ =='__main__':
    import csv

    with open('test.in', 'r') as input_file:
        # 逐行读取输入文件
        input_lines = input_file.readlines()

    with open('output.csv', 'w', newline='') as output_file:
        # 创建 CSV 写入器并写入表头数据
        writer = csv.writer(output_file)
        writer.writerow(['id', 'student_id', 'name', 'department', 'course'])

        # 逐行解析数据并写入 CSV 文件
        i = 0
        while i < len(input_lines):
            # 读取每个学生的id值
            id_value = input_lines[i].strip()
            i += 1

            # 读取学生相关信息
            student_id = input_lines[i].strip()
            i += 1

            name = input_lines[i].strip()
            i += 1

            department = input_lines[i].strip()
            i += 1

            course = input_lines[i].strip()
            i += 1

            # 将该学生的信息写入 CSV 文件
            writer.writerow([id_value, student_id, name, department, course])

    # 关闭输入和输出文件
    input_file.close()
    output_file.close()