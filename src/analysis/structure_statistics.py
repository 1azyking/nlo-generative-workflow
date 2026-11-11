import os
from pymatgen.core.structure import Structure, Element
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


def analyze_cif_files(folder="."):
    """
    Analyze the element count and site count distributions in CIF files in the given folder.

    Parameters:
    folder (str): Path to the folder containing CIF files. Default is the current directory.

    Returns:
    None
    """
    element_counts = []
    site_counts = []

    # Iterate through all files in the folder
    for file in os.listdir(folder):
        if file.endswith(".cif"):  # Process only CIF files
            try:
                structure = Structure.from_file(os.path.join(folder, file))
                element_counts.append(len(structure.composition.elements))  # Count unique elements
                site_counts.append(len(structure))  # Count total sites
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # Compute distributions
    element_count_distribution = Counter(element_counts)
    site_count_distribution = Counter(site_counts)

    # Print results
    print("num_elems\tnum_structs")
    for count, freq in sorted(element_count_distribution.items()):
        print(f"{count}\t{freq}")

    print("num_sites\tnum_structs")
    for count, freq in sorted(site_count_distribution.items()):
        print(f"{count}\t{freq}")


TARGET_ELEMENTS = {"B", "C", "N", "Si", "P", "S"}


def count_target_and_metal_cations(folder="."):
    """
    Analyze CIF files to count the number of target elements and metal cations in each structure.

    Parameters:
    folder (str): Path to the folder containing CIF files. Default is the current directory.

    Returns:
    None
    """
    results = []

    for file in os.listdir(folder):
        if file.endswith(".cif"):
            try:
                structure = Structure.from_file(os.path.join(folder, file))
                composition = structure.composition

                # 统计目标元素种类数量
                target_elements = [el for el in composition.elements if el.symbol in TARGET_ELEMENTS]
                num_target_elements = len(target_elements)

                # 统计金属阳离子种类数量（根据金属性质判断）
                metal_cations = [el for el in composition.elements if Element(el).is_metal]
                num_metal_cations = len(metal_cations)

                results.append({
                    "file": file,
                    "num_target_elements": num_target_elements,
                    "num_metal_cations": num_metal_cations
                })
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # 打印结果
    print(f"{'File':<30}{'Target Elements':<20}{'Metal Cations':<20}")
    print("-" * 70)
    for res in results:
        print(f"{res['file']:<30}{res['num_target_elements']:<20}{res['num_metal_cations']:<20}")


def summarize_cif_statistics(folder="."):
    """
    Summarize the total number of unique target elements and metal cations in all CIF files.

    Parameters:
    folder (str): Path to the folder containing CIF files. Default is the current directory.

    Returns:
    None
    """
    all_target_elements = set()
    all_metal_cations = set()

    for file in os.listdir(folder):
        if file.endswith(".cif"):
            try:
                # 读取结构文件
                structure = Structure.from_file(os.path.join(folder, file))
                composition = structure.composition

                # 收集目标元素
                for el in composition.elements:
                    if el.symbol in TARGET_ELEMENTS:
                        all_target_elements.add(el.symbol)

                # 收集金属阳离子
                for el in composition.elements:
                    if Element(el).is_metal:
                        all_metal_cations.add(el.symbol)
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # 输出统计结果
    print("Summary Statistics:")
    print(f"Total unique target elements ({', '.join(TARGET_ELEMENTS)}): {len(all_target_elements)}")
    print(f"Target elements found: {', '.join(sorted(all_target_elements)) if all_target_elements else 'None'}")
    print(f"Total unique metal cations: {len(all_metal_cations)}")
    print(f"Metal cations found: {', '.join(sorted(all_metal_cations)) if all_metal_cations else 'None'}")


def count_element_distribution(folder="."):
    """
    Count the distribution of structures based on the number of target elements they contain.

    Parameters:
    folder (str): Path to the folder containing CIF files. Default is the current directory.

    Returns:
    None
    """
    target_count_distribution = {}

    for file in os.listdir(folder):
        if file.endswith(".cif"):
            try:
                # 读取 CIF 文件的结构
                structure = Structure.from_file(os.path.join(folder, file))
                composition = structure.composition

                # 统计含有的目标元素种类数
                target_elements_in_structure = {el.symbol for el in composition.elements if el.symbol in TARGET_ELEMENTS}
                # 统计含有的金属元素种类数
                # target_elements_in_structure = {el.symbol for el in composition.elements if el.is_metal}

                num_target_elements = len(target_elements_in_structure)

                # 更新统计分布
                target_count_distribution[num_target_elements] = target_count_distribution.get(num_target_elements, 0) + 1
                # if num_target_elements == 0:
                #     print(file[:-4])
            except Exception as e:
                print(f"Error processing {file}: {e}")

    # 输出统计结果
    print("Distribution of structures based on the number of target elements:")
    for num_elements, count in sorted(target_count_distribution.items()):
        print(f"{num_elements}\t{count}")


if __name__ == "__main__":
    dataset_file = "bcn_40_gens"
    # dataset_file = "5sets_data"
    count_element_distribution(f"C:/Users/13701/Desktop/gm_cifs/cdvae/v32/to_plot/{dataset_file}/")
