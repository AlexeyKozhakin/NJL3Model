import random

def generate_random_permutation(n, k):
    """
    Функция генерации случайной перестановки длиной n из k целых чисел
    Если n>k, то в перестановке возникают повоторения 
    """

    permutation = list(range(1, k+1))

    random.shuffle(permutation)
    # Если k больше n, добавляем дополнительные числа от n + 1 до k
    if n > k:
        permutation = permutation*(n//k)+permutation[:n%k]
#         permutation.extend([k-n])

    # Перемешиваем список случайным образом
    random.shuffle(permutation)
    print(permutation)
    return permutation
    
    
def count_numbers_between(a, b):
    """
    Функция для подсчета количества целых чисел между a и b (не включая их самих).
    """
    count = abs(b - a) + 1
    return max(0, count)

def count_unique_values(n,params):
  k={}
  for name, space in params.items():
    if space[-1]=='float':
      k[name] = n
    if space[-1]=='int':
      k[name] = count_numbers_between(space[0][0], space[0][1])
    if space[-1]=='cat':
      k[name] = len(space[0])
  return k

def calculateTotalCalculations(n,params):
    k = count_unique_values(n,params)
    Nall = 1
    for size in k.values():
      Nall*=size
    return Nall
    
    
def generate_permutation_matrix(n, k):
    matrix = {}
    for name, size in k.items():
#        k = count_unique_values(param[p])
        matrix[name]=generate_random_permutation(n, size)  # Добавляем перестановку в матрицу
    return matrix
    
    
def generate_parameter_sample_values(n, params, random_state=1):
  random.seed(random_state)
  parameter_sample_values = {}
  k = count_unique_values(n,params)
  permutation_matrix = generate_permutation_matrix(n, k)
  for name, space in params.items():
    for i in permutation_matrix[name]:
      if space[-1]=='float':
         if name in parameter_sample_values:
           parameter_sample_values[name].append(space[0][0]+(i-1)*(space[0][1]-space[0][0])/(n-1))
         else:
           parameter_sample_values[name] = [space[0][0]+(i-1)*(space[0][1]-space[0][0])/(n-1)]
      elif space[-1]=='int':
         if name in parameter_sample_values:
           parameter_sample_values[name].append(space[0][0]+(i-1))
         else:
           parameter_sample_values[name] = [space[0][0]+(i-1)]
      elif space[-1]=='cat':
         if name in parameter_sample_values:
           parameter_sample_values[name].append(space[0][i-1])
         else:
           parameter_sample_values[name] = [space[0][(i-1)]]           
  return parameter_sample_values


def fmin(fn, params, verbose = True):
    """
    Функция для нахождения минимума целевой функции target_function
    среди всех комбинаций параметров, заданных в parameters.

    Аргументы:
    fn: функция, которую нужно минимизировать.
                     Принимает словарь с параметрами в качестве аргумента.
    parames: словарь параметров, где ключ - имя параметра,
                значение - список значений параметра.

    Возвращает:
    best_params: словарь лучших параметров, при которых целевая функция минимальна.
    best_value: значение целевой функции при лучших параметрах.
    """

    best_value = float('inf')  # Начальное значение лучшего результата
    best_params = None

    # Проходим по всем комбинациям параметров
    current_params={}
    for row_index in range(len(params.values())):
      # Допустим, нам нужна первая строка (индекс 0)
        # Получаем нужную строку из таблицы
        current_params = {key: values[row_index] for key, values in params.items()}
        current_value = fn(current_params)
        if verbose:
          print(f'metric is {current_value}')

        # Если текущее значение лучше предыдущего минимума, обновляем лучшие параметры и значение
        if current_value < best_value:
            best_value = current_value
            best_params = current_params

    return best_params, best_value