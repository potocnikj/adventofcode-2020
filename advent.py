import itertools
from typing import List
from typing import Dict
from typing import Tuple
import time


class Puzzle:
    _FILES = {
        1: 'puzzle1.txt',
        2: 'puzzle2.txt',
        3: 'puzzle3.txt',
        4: 'puzzle4.txt',
        5: 'puzzle5.txt',
        6: 'puzzle6.txt',
        7: 'puzzle7.txt'
    }

    def __init__(self):
        self.solve()

    def solve(self):
        pass

    @staticmethod
    def _get_data_from_file(file: str) -> List[str]:
        """
        A private method that reads all lines from requested file, casts them into a list
        and returns it.
        """
        with open(file) as file_in:
            lines = []
            for line in file_in:
                lines.append(line.replace('\n', ''))
            return lines

    @staticmethod
    def _get_raw_data_from_file(file: str) -> str:
        with open(file) as file_in:
            data = file_in.read()
        return data

    @staticmethod
    def _print_results(puzzle: int, task: int, result, time_spent: float):
        """
        Prints results to console, takes puzzle and task number into account
        """
        duration = round(time_spent * 1000, 2)
        print('[*] Task: {0} Part: {1} Result::: {2} Duration: {3} ms'.format(puzzle, task, result, duration)
              )


class FirstPuzzle(Puzzle):
    @classmethod
    def solve(cls):
        data = cls._get_data_from_file(cls._FILES[1])
        cls._solve(data, 2)
        cls._solve(data, 3)

    @classmethod
    def _solve(cls, data: List[str], combination_order: int):
        start = time.time()
        serialized_data = []
        for i in data:
            serialized_data.append(int(i))
        pairs = list(itertools.combinations(data, combination_order))

        result = None
        for a in pairs:
            s = 0
            for el in a:
                s += int(el)
            if s == 2020:
                result = 1
                for el in a:
                    result *= int(el)
        end = time.time()
        cls._print_results(1, combination_order - 1, result, (end - start))


class SecondPuzzle(Puzzle):
    @classmethod
    def solve(cls):
        data = cls._get_data_from_file(cls._FILES[2])
        serialized_data = []
        for el in data:
            components = el.split(' ')
            if len(components) != 3:
                raise Exception('Invalid data component!')

            length = components[0].split('-')
            if (len(length)) != 2:
                raise Exception('Invalid length data')

            int_len = []
            for i in length:
                int_len.append(int(i))
            serialized_data.append(
                {'password': components[2], 'key': components[1].replace(':', ''), 'min_len': int(min(int_len)),
                 'max_len': int(max(int_len))}
            )
        cls._first_part(serialized_data)
        cls._second_part(serialized_data)

    @classmethod
    def _first_part(cls, data: List[Dict]):
        start = time.time()
        number_of_valid_passwords = 0
        for i in data:
            number_of_occurrences = i['password'].count(i['key'])
            if i['min_len'] <= number_of_occurrences <= i['max_len']:
                number_of_valid_passwords += 1

        end = time.time()
        cls._print_results(2, 1, number_of_valid_passwords, (end - start))

    @classmethod
    def _second_part(cls, data: List[Dict]):
        start = time.time()
        number_of_valid_passwords = 0
        for i in data:
            min_key = i['min_len'] - 1
            max_key = i['max_len'] - 1
            max_found = False
            min_found = False
            if min_key >= 0 and min_key < len(i['password']):
                if i['password'][min_key] == i['key']:
                    min_found = True

            if max_key >= 0 and max_key < len(i['password']):
                if i['password'][max_key] == i['key']:
                    max_found = True

            if max_found and not min_found:
                number_of_valid_passwords += 1
            elif min_found and not max_found:
                number_of_valid_passwords += 1

        end = time.time()
        cls._print_results(2, 2, number_of_valid_passwords, (end - start))


class ThirdPuzzle(Puzzle):
    @classmethod
    def solve(cls):
        data = cls._get_data_from_file(cls._FILES[3])
        cls._first_part(data)
        cls._second_part(data)

    @classmethod
    def _first_part(cls, data: List[str]):
        start = time.time()
        map_width = 0  # Width is the len of every string in list, and all of them must have the same length!
        map_depth = len(data)  # Length represents depth of our map
        for i in data:
            if map_width == 0:
                map_width = len(i)
                continue
            if map_width != 0 and len(i) != map_width:
                raise Exception('Invalid data, map does not appear to be in rectangular form!')

        factor = (map_depth / map_width + 1) * 3  # We need to generate a map, wide enough for traversal.
        extended_data = []
        for i in data:
            extended_data.append(i * int(factor))

        horizontal = 3
        number_of_trees = 0
        for i in range(1, len(extended_data)):
            if extended_data[i][horizontal] == '#':  # We found a tree!
                number_of_trees += 1
            horizontal += 3

        end = time.time()
        cls._print_results(3, 1, number_of_trees, (end - start))

    @classmethod
    def _second_part(cls, data: List[str]):
        start = time.time()
        traversals = [(1, 1), (3, 1), (5, 1), (7, 1), (1, 2)]
        map_width = 0  # Width is the len of every string in list, and all of them must have the same length!
        map_depth = len(data)  # Length represents depth of our map
        for i in data:
            if map_width == 0:
                map_width = len(i)
                continue
            if map_width != 0 and len(i) != map_width:
                raise Exception('Invalid data, map does not appear to be in rectangular form!')

        factor = (map_depth / map_width + 1) * 10  # We need to generate a map, wide enough for traversal.
        extended_data = []
        for i in data:
            extended_data.append(i * int(factor))

        number_of_trees = []
        for r, d in traversals:
            horizontal = r
            trees = 0
            for i in range(d, len(extended_data), d):
                if extended_data[i][horizontal] == '#':  # We found a tree!
                    trees += 1
                horizontal += r
            number_of_trees.append(trees)

        result = 1
        for i in number_of_trees:
            result = result * i

        end = time.time()
        cls._print_results(3, 2, result, (end - start))


class FourthPuzzle(Puzzle):
    @classmethod
    def solve(cls):
        required_passport_fields = {'byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid', 'cid'}
        # It's just fine if we're missing cid field
        north_pole_fields = {'byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid'}
        data = cls._get_data_from_file(cls._FILES[4])
        # As data is separated not by new lines but rather whitespaces, we need to fix that
        serialized_data = []
        passport = ''
        for i in data:
            if i == '':
                serialized_data.append(passport)
                passport = ''
            else:
                passport = passport + ' ' + i
        serialized_data.append(passport)  # Must not forget the last one! :)
        cls._first_part(serialized_data, required_passport_fields, north_pole_fields)
        cls._second_part(serialized_data, required_passport_fields, north_pole_fields)

    @classmethod
    def _first_part(cls, data, passport_fields, north_pole_fields):
        start = time.time()
        passport_keys = []
        for i in data:
            first_split = i.split(' ')
            keys = []
            for j in first_split:
                k = j.split(':')[0]
                if k != '':
                    keys.append(k)
            passport_keys.append(keys)

        number_of_valid_passports = 0
        for i in passport_keys:
            if set(i) == passport_fields or set(i) == north_pole_fields:
                number_of_valid_passports += 1

        end = time.time()
        cls._print_results(4, 1, number_of_valid_passports, (end - start))

    @classmethod
    def _second_part(cls, data, passport_fields, north_pole_fields):
        start = time.time()
        passport_keys = []
        for i in data:
            first_split = i.split(' ')
            keys = []
            for j in first_split:
                spl = j.split(':')
                if spl[0] != '':
                    keys.append((spl[0], spl[1]))
            passport_keys.append(keys)

        # We only need those that match keys criteria in the first place
        filtered = []
        for i in passport_keys:
            k = [j[0] for j in i]
            if set(k) == passport_fields or set(k) == north_pole_fields:
                filtered.append(i)

        eye_colors = ['amb', 'blu', 'brn', 'gry', 'grn', 'hzl', 'oth']
        number_of_valid_passports = 0
        for j in filtered:
            valid_props = 0
            for p in j:
                if p[0] == 'byr':
                    try:
                        if int(p[1]) <= 2002 and int(p[1]) >= 1920:
                            valid_props += 1
                    except:
                        pass
                if p[0] == 'iyr':
                    try:
                        if int(p[1]) <= 2020 and int(p[1]) >= 2010:
                            valid_props += 1
                    except:
                        pass
                if p[0] == 'eyr':
                    try:
                        if int(p[1]) <= 2030 and int(p[1]) >= 2020:
                            valid_props += 1
                    except:
                        pass
                if p[0] == 'hgt':
                    fp = p[1][:-2]
                    sp = p[1][-2::]
                    if sp == 'cm' and int(fp) >= 150 and int(fp) <= 193:
                        valid_props += 1
                    elif sp == 'in' and int(fp) >= 59 and int(fp) <= 76:
                        valid_props += 1
                if p[0] == 'hcl':
                    valid = True
                    if len(p[1]) != 7 or p[1][0] != '#':
                        valid = False
                    for c in p[1][1::]:
                        if c not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']:
                            valid = False
                    if valid:
                        valid_props += 1
                if p[0] == 'ecl':
                    if p[1] in eye_colors:
                        valid_props += 1
                if p[0] == 'pid':
                    valid = True
                    if len(p[1]) != 9:
                        valid = False
                    for c in p[1]:
                        if c not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                            valid = False
                    if valid:
                        valid_props += 1
            if valid_props == 7:
                number_of_valid_passports += 1

        end = time.time()
        cls._print_results(4, 2, number_of_valid_passports, (end - start))


class FifthPuzzle(Puzzle):

    @classmethod
    def solve(cls):
        start = time.time()
        data = cls._get_data_from_file(cls._FILES[5])
        seat_positions = []
        for i in data:
            row_span = [i for i in range(128)]
            col_span = [i for i in range(8)]
            # first find appropriate row
            for j in range(0, 7):
                half = len(row_span) // 2
                if i[j] == 'F':
                    row_span = row_span[:half]
                if i[j] == 'B':
                    row_span = row_span[half:]
            # Then find appropriate column
            for j in range(7, 10):
                half = len(col_span) // 2
                if i[j] == 'L':
                    col_span = col_span[:half]
                if i[j] == 'R':
                    col_span = col_span[half:]
            seat_positions.append((row_span[0], col_span[0]))

        # calculate maximum seatID

        seat_ids = []
        for row, col in seat_positions:
            seat_ids.append((row * 8) + col)

        end = time.time()
        cls._first_part(seat_ids, (end - start))
        cls._second_part(seat_positions, start)

    @classmethod
    def _first_part(cls, seat_ids: List[int], time_spent: float):
        cls._print_results(5, 1, max(seat_ids), time_spent)

    @classmethod
    def _second_part(cls, seat_positions: List[Tuple[int, int]], start: float):
        sorted_positions = sorted(seat_positions)
        rows = []
        for j, _ in sorted_positions:
            rows.append(j)
        rows = list(set(rows))
        first_row = min(rows)
        last_row = max(rows)
        row_dict = {i: [] for i in rows}
        for i, j in sorted_positions:
            row_dict[i].append(j)

        # We need to find the row that hasn't got all of the columns!
        our_seat = {
            'row': 0,
            'column': 0
        }
        missing_seats = 0
        for i, j in row_dict.items():
            # We know (because of instructions) that our seat is not in the first or last row, hence we skip it
            if i == first_row or i == last_row:
                continue
            if len(j) != 8:
                missing_seats += 1
                if len(j) != 7:
                    raise Exception('There is more than one column missing - Probably our miscalculation!')
                our_seat['row'] = i
                # Now find which column is missing
                for c in [0, 1, 2, 3, 4, 5, 6, 7]:
                    if c not in j:
                        our_seat['column'] = c

        if missing_seats != 1:
            raise Exception('There is more than one seat missing - Probably our miscalculation')

        our_seat_id = our_seat['row'] * 8 + our_seat['column']
        end = time.time()
        cls._print_results(5, 2, our_seat_id, (end - start))


class SixthPuzzle(Puzzle):

    @classmethod
    def solve(cls):
        data = cls._get_raw_data_from_file(cls._FILES[6]).split('\n\n')
        cls._first_part(data)
        cls._second_part(data)

    @classmethod
    def _first_part(cls, data: List[str]):
        start = time.time()
        result = 0
        for group in data:
            result += len(set(group.replace('\n', '')))
        end = time.time()
        cls._print_results(6, 1, result, (end - start))

    @classmethod
    def _second_part(cls, data: List[str]):
        data[len(data) - 1] = data[len(data) - 1][:-1]  # Remove annoying trailing line break
        start = time.time()
        result = 0
        for group in data:
            result += len(set.intersection(*[set(a) for a in group.split('\n')]))

        end = time.time()
        cls._print_results(6, 2, result, (end - start))


class SeventhPuzzle(Puzzle):

    @classmethod
    def solve(cls):
        data = cls._get_data_from_file(cls._FILES[7])
        start = time.time()
        serialized_data = []
        for i in data:
            if 'shiny gold' not in i[0]:
                serialized_data.append([bag.replace('bags', '').replace('bag', '') for bag in
                                        i.replace('.', '').replace(',', '').split(' bags contain ') if bag != ''])

        cls._first_part(serialized_data, start)
        cls._second_part(serialized_data, start)

    @classmethod
    def _first_part(cls, serialized_data, start: float):
        # We now have our data structure prepared for building a corresponding tree

        # Serialize the data so we can work with it as with tree structure
        valid_bags = 0
        clean_data = {}
        for i in serialized_data:
            new_stmts = []
            for statement in i:
                new_stmt = ''
                for char in statement:
                    if not char.isnumeric():
                        new_stmt += char
                    else:
                        new_stmt += '*'
                new_stmts.append([el.replace(' ', '') for el in new_stmt.split('*') if el != ''])

            if len(new_stmts) != 2:
                raise Exception('Our data structure is not formed right!')

            clean_data[new_stmts[0][0]] = new_stmts[1]
        # On initial step, we're going to find bags that eventually include our bag
        all_bags = []
        nodes_with_our_bag = ['shinygold']
        while len(nodes_with_our_bag) > 0:
            new_nodes = []
            for bag in nodes_with_our_bag:
                for node, children in clean_data.items():
                    if bag in children:
                        new_nodes.append(node)
            nodes_with_our_bag = list(set(new_nodes))
            all_bags += nodes_with_our_bag

        end = time.time()
        cls._print_results(7, 1, len(list(set(all_bags))), (end - start))

    @classmethod
    def _second_part(cls, serialized_data, start: float):
        clean_data = {}
        for line in serialized_data:
            numbered_lines = line[1].replace('  ', ',').replace(' ', '').split(',')
            if numbered_lines == ['noother']:
                clean_data[line[0].replace(' ', '')] = None
            else:
                clean_data[line[0].replace(' ', '')] = {el[1::]: int(el[0]) for el in numbered_lines}
        tree = clean_data['shinygold']
        result = cls._traverse_tree(tree, clean_data)
        end = time.time()
        cls._print_results(7, 2, result, (end - start))

    @classmethod
    def _traverse_tree(cls, tree: Dict, clean_data: Dict):
        result = 0
        for node, amount in tree.items():
            if clean_data[node] is not None:
                result += amount + amount * cls._traverse_tree(clean_data[node], clean_data)
            else:
                result += amount
        return result


FirstPuzzle()
SecondPuzzle()
ThirdPuzzle()
FourthPuzzle()
FifthPuzzle()
SixthPuzzle()
SeventhPuzzle()
