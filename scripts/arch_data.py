import pickle
from os.path import expanduser


class ArchData:
    def __init__(self):
        self.arch_id = None
        self.train_search_id = None
        self.best_train_id = None
        self.super_model_acc = None
        self.model_acc = None
        self.time_for_100_inf = None
        self.macs_count = None
        self.genotype_txt = None
        self.closs_w = None
        self.git_hash = None


class ArchDataCollection:
    def __init__(self, collection_file_path='{}/.arch_data'.format(expanduser('~'))):
        self.collection_file_path = collection_file_path
        self.archs = None

    def save(self):
        with open(self.collection_file_path, 'wb') as fp:
            pickle.dump(self.archs, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        try:
            with open(self.collection_file_path, 'rb') as fp:
                self.archs = pickle.load(fp)
        except FileNotFoundError:
            self.archs = {}

    def add_arch(self, arch_id, train_search_id, best_train_id=None):
        if arch_id not in self.archs:
            a = ArchData()
            a.arch_id = arch_id
            a.train_search_id = train_search_id
            self.archs[arch_id] = a
        else:
            a = self.archs[arch_id]
        a.best_train_id = best_train_id

    def csv_dump(self, path):
        with open(path, 'w') as fp:
            fp.write('{}, {}, {}, {}, {}, {}, {}, {}\n'.format('arch_id', 'train_search_id', 'best_train_id',
                                                               'super_model_acc', 'model_acc', 'time_for_100_inf',
                                                               'macs_count', 'genotype_txt'))
            for arch in self.archs.values():
                fp.write('{}, {}, {}, {}, {}, {}, {}, "{}"\n'.format(arch.arch_id, arch.train_search_id,
                                                                     arch.best_train_id, arch.super_model_acc,
                                                                     arch.model_acc, arch.time_for_100_inf,
                                                                     arch.macs_count, arch.genotype_txt))


def create_update_arch_collection():
    arch_c = ArchDataCollection()
    arch_c.load()
    arch_c.add_arch('M2', 'search-EXP-20210904-015723', 'eval-EXP-20211001-182408')
    arch_c.add_arch('M3', 'search-EXP-20210909-083612', 'eval-EXP-20210929-092446')
    arch_c.add_arch('M5', 'search-EXP-20210914-132854', 'eval-EXP-20210930-001834')
    arch_c.add_arch('M6', 'search-EXP-20210915-193742', 'eval-EXP-20210930-012047')
    arch_c.add_arch('M7', 'search-EXP-20210916-134929', 'eval-EXP-20210930-144310')
    arch_c.add_arch('M8', 'search-EXP-20210923-085601', 'eval-EXP-20211018-175456')
    arch_c.add_arch('M9', 'search-EXP-20210930-113256', 'eval-EXP-20211019-000907')
    arch_c.add_arch('M10', 'search-EXP-20211007-153233', 'eval-EXP-20211008-102216')
    arch_c.add_arch('M11', 'search-EXP-20211008-090653', 'eval-EXP-20211019-013549')
    arch_c.add_arch('M12', 'search-EXP-20211025-164301', 'eval-EXP-20211028-153142')
    arch_c.add_arch('M13', 'search-EXP-20211027-172023', '')
    arch_c.add_arch('M14', 'search-EXP-20211028-140821', '')
    arch_c.add_arch('M15', 'search-EXP-20211029-090036', '')
    arch_c.add_arch('M16', 'search-EXP-20211101-194529', '')
    arch_c.add_arch('M17', 'search-EXP-20211101-195209', '')
    arch_c.add_arch('M19', 'search-EXP-20211103-234754', '')
    arch_c.add_arch('M20', 'search-EXP-20211105-120555', '')
    arch_c.add_arch('M21', 'search-EXP-20211105-143218', '')
    arch_c.add_arch('M22', 'search-EXP-20211106-152513', '')
    arch_c.add_arch('M23', 'search-EXP-20211106-152514', '')
    arch_c.add_arch('M24', 'search-EXP-20211107-092623', '')
    arch_c.add_arch('M25', 'search-EXP-20211107-092453', '')
    arch_c.add_arch('M26', 'search-EXP-20211107-233004', '')
    arch_c.add_arch('M27', 'search-EXP-20211107-233603', '')
    arch_c.add_arch('M28', 'search-EXP-20211109-101420', '')
    arch_c.add_arch('M29', 'search-EXP-20211109-102212', '')
    arch_c.add_arch('M30', 'search-EXP-20211111-091754', '')
    arch_c.add_arch('M31', 'search-EXP-20211111-093439', '')
    arch_c.add_arch('M32', 'search-EXP-20211111-193703', '')
    arch_c.add_arch('M33', 'search-EXP-20211111-195217', '')
    arch_c.add_arch('M34', 'search-EXP-20211112-091217', '')
    arch_c.add_arch('M35', 'search-EXP-20211112-065726', '')
    arch_c.add_arch('M36', 'search-EXP-20211117-161757', '')
    arch_c.add_arch('M37', 'search-EXP-20211117-234131', '')
    arch_c.add_arch('M38', 'search-EXP-20211117-234203', '')
    arch_c.add_arch('M39', 'search-EXP-20211118-113344', '')
    arch_c.add_arch('M40', 'search-EXP-20211118-115241', '')
    arch_c.add_arch('M41', 'search-EXP-20211118-210858', '')
    arch_c.add_arch('M42', 'search-EXP-20211118-211000', '')
    arch_c.add_arch('M43', 'search-EXP-20211122-124041', '')
    arch_c.add_arch('M44', 'search-EXP-20211122-124108', '')
    arch_c.add_arch('M45', 'search-EXP-20211123-144344', '')
    arch_c.add_arch('M46', 'search-EXP-20211123-144732', '')
    arch_c.add_arch('M47', 'search-EXP-20211124-021521', '')
    arch_c.add_arch('M48', 'search-EXP-20211124-094008', '')
    arch_c.add_arch('M49', 'search-EXP-20211124-113043', '')
    arch_c.add_arch('M50', 'search-EXP-20211124-215136', '')
    arch_c.add_arch('M51', 'search-EXP-20211124-215518', '')
    arch_c.add_arch('M52', 'search-EXP-20211125-101248', '')
    arch_c.add_arch('M53', 'search-EXP-20211125-101656', '')
    arch_c.add_arch('M54', 'search-EXP-20211125-200349', '')
    arch_c.add_arch('M55', 'search-EXP-20211125-200624', '')
    arch_c.add_arch('M56', 'search-EXP-20211129-161051', '')
    arch_c.add_arch('M57', 'search-EXP-20211129-161227', '')
    arch_c.add_arch('M58', 'search-EXP-20211130-093027', '')
    arch_c.add_arch('M59', 'search-EXP-20211130-094257', '')
    arch_c.add_arch('M60', 'search-EXP-20211130-191429', '')
    arch_c.add_arch('M61', 'search-EXP-20211130-191926', '')
    arch_c.save()
