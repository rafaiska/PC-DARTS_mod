import pickle
from enum import Enum
from os.path import expanduser


class CLossV(Enum):
    ORIGINAL = 1  # Original PC-DARTS Loss Function (pure Cross-Entropy Loss)
    LEGACY = 2  # Experiments prior to differentiable custom loss
    D_LOSS_V1 = 3  # op_oracle with MAC based weights, differentiable op_loss v1
    D_LOSS_V2 = 4  # op_oracle with MAC based weights, differentiable op_loss v2 (with reduce importance)
    D_LOSS_V3 = 5  # op_oracle with MAC based weights, differentiable op_loss v3 (adjustment on zero MACS operators)
    D_LOSS_V4 = 6  # Same as M55, but without using different criterions for arch and regular optimizers


class UsedGPU(Enum):
    SDUMONT_K40 = 1
    CENAPAD_A100 = 2


def quotes(s):
    return '"' + s + '"'


class ArchData:
    ARCH_CSV_HEADER = ['arch_id', 'train_search_id', 'best_train_id',
                       'super_model_acc', 'model_acc', 'time_for_100_inf',
                       'macs_count', 'genotype_txt', 'closs_v', 'closs_w']

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
        self.closs_v = None
        self.git_hash = None
        self.train_search_gpu = None
        self.best_train_gpu = None

    def __str__(self):
        return quotes('","'.join([str(self.__getattribute__(k)) for k in ArchData.ARCH_CSV_HEADER]))

    @staticmethod
    def get_csv_header():
        return quotes('","'.join(ArchData.ARCH_CSV_HEADER))


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

    def add_arch(self, arch_id, train_search_id, best_train_id=None, closs_v=None):
        if arch_id not in self.archs:
            a = ArchData()
            a.arch_id = arch_id
            a.train_search_id = train_search_id
            a.closs_v = closs_v
            self.archs[arch_id] = a
        else:
            a = self.archs[arch_id]
        a.best_train_id = best_train_id

    def csv_dump(self, path):
        with open(path, 'w') as fp:
            fp.write('{}\n'.format(ArchData.get_csv_header()))
            for arch in self.archs.values():
                fp.write('{}\n'.format(str(arch)))


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
    arch_c.add_arch('M15', 'search-EXP-20211029-090036', 'eval-EXP-20211220-000805')
    arch_c.add_arch('M16', 'search-EXP-20211101-194529', '')
    arch_c.add_arch('M17', 'search-EXP-20211101-195209', 'eval-EXP-20211220-175233')
    arch_c.add_arch('M19', 'search-EXP-20211103-234754', 'eval-EXP-20211221-034230')
    arch_c.add_arch('M20', 'search-EXP-20211105-120555', '')
    arch_c.add_arch('M21', 'search-EXP-20211105-143218', 'eval-EXP-20211221-144349')
    arch_c.add_arch('M22', 'search-EXP-20211106-152513', '')
    arch_c.add_arch('M23', 'search-EXP-20211106-152514', 'eval-EXP-20211221-153947')
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
    arch_c.add_arch('M45', 'search-EXP-20211123-144344', 'eval-EXP-20211208-022915')
    arch_c.add_arch('M46', 'search-EXP-20211123-144732', 'eval-EXP-20211208-084129')
    arch_c.add_arch('M47', 'search-EXP-20211124-021521', 'eval-EXP-20211208-151703')
    arch_c.add_arch('M48', 'search-EXP-20211124-094008', 'eval-EXP-20211209-073810')
    arch_c.add_arch('M49', 'search-EXP-20211124-113043', 'eval-EXP-20211209-153606')
    arch_c.add_arch('M50', 'search-EXP-20211124-215136', 'eval-EXP-20211210-072834')
    arch_c.add_arch('M51', 'search-EXP-20211124-215518', 'eval-EXP-20211210-203424')
    arch_c.add_arch('M52', 'search-EXP-20211125-101248', 'eval-EXP-20211211-132059')
    arch_c.add_arch('M53', 'search-EXP-20211125-101656', 'eval-EXP-20211212-042820')
    arch_c.add_arch('M54', 'search-EXP-20211125-200349', 'eval-EXP-20211213-004956')
    arch_c.add_arch('M55', 'search-EXP-20211125-200624', 'eval-EXP-20211213-105229')
    arch_c.add_arch('M56', 'search-EXP-20211129-161051', 'eval-EXP-20211201-111607')
    arch_c.add_arch('M57', 'search-EXP-20211129-161227', 'eval-EXP-20211201-111005')
    arch_c.add_arch('M58', 'search-EXP-20211130-093027', 'eval-EXP-20211213-163128')
    arch_c.add_arch('M59', 'search-EXP-20211130-094257', 'eval-EXP-20211222-054906')
    arch_c.add_arch('M60', 'search-EXP-20211130-191429', 'eval-EXP-20211214-212312')
    arch_c.add_arch('M61', 'search-EXP-20211130-191926', 'eval-EXP-20220117-171813')
    arch_c.add_arch('M62', 'search-EXP-20211213-171055-0', 'eval-EXP-20211222-141731')
    arch_c.add_arch('M63', 'search-EXP-20211213-171054-1', 'eval-EXP-20211222-232025')
    arch_c.add_arch('M64', 'search-EXP-20211213-171054-3', 'eval-EXP-20211223-052635')
    arch_c.add_arch('M65', 'search-EXP-20211214-200624-0', 'eval-EXP-20211223-103532')
    arch_c.add_arch('M66', 'search-EXP-20211214-200620-0', 'eval-EXP-20211223-113043')
    arch_c.add_arch('M67', 'search-EXP-20211214-200618-1', 'eval-EXP-20211223-211408')
    arch_c.add_arch('M68', 'search-EXP-20211214-200620-2', 'eval-EXP-20211224-032740')
    arch_c.add_arch('M69', 'search-EXP-20211214-200623-2', 'eval-EXP-20211224-111843')
    arch_c.add_arch('M70', 'search-EXP-20211214-200622-3', 'eval-EXP-20211224-181625')
    arch_c.add_arch('M71', 'search-EXP-20211214-200620-3', 'eval-EXP-20211225-042043')
    arch_c.add_arch('M72', 'search-EXP-20211216-051706-1', 'eval-EXP-20211225-053629')
    arch_c.add_arch('M73', 'search-EXP-20211216-051705-2', 'eval-EXP-20211225-105923')
    arch_c.add_arch('M74', 'search-EXP-20211216-051658-2', 'eval-EXP-20211226-055411')
    arch_c.add_arch('M75', 'search-EXP-20211216-051706-3', 'eval-EXP-20220118-230414')
    arch_c.add_arch('M76', 'search-EXP-20211216-051706-0', 'eval-EXP-20211227-074753')
    arch_c.add_arch('M77', 'search-EXP-20211216-170404-0', 'eval-EXP-20220118-231404', CLossV.D_LOSS_V4)
    arch_c.add_arch('M78', 'search-EXP-20211216-170410-1', 'eval-EXP-20211229-234640', CLossV.D_LOSS_V4)
    arch_c.add_arch('M79', 'search-EXP-20211216-170405-1', 'eval-EXP-20211230-061455', CLossV.D_LOSS_V4)
    arch_c.add_arch('M80', 'search-EXP-20211216-170404-2', 'eval-EXP-20211231-162039', CLossV.D_LOSS_V4)
    arch_c.add_arch('M81', 'search-EXP-20211216-170409-2', 'eval-EXP-20220101-072954', CLossV.D_LOSS_V4)
    arch_c.add_arch('M82', 'search-EXP-20211216-170407-3', 'eval-EXP-20220101-164450', CLossV.D_LOSS_V4)
    arch_c.add_arch('M83', 'search-EXP-20211216-170404-3', 'eval-EXP-20220101-222206', CLossV.D_LOSS_V4)
    arch_c.add_arch('M84', 'search-EXP-20211217-113311-0', 'eval-EXP-20220103-011146', CLossV.D_LOSS_V4)
    arch_c.add_arch('M85', 'search-EXP-20211217-113327-0', 'eval-EXP-20220103-073434', CLossV.D_LOSS_V4)
    arch_c.add_arch('M86', 'search-EXP-20211217-113328-1', 'eval-EXP-20220103-085104', CLossV.D_LOSS_V4)
    arch_c.add_arch('M87', 'search-EXP-20211217-113312-1', 'eval-EXP-20220103-170705', CLossV.D_LOSS_V4)
    arch_c.add_arch('M88', 'search-EXP-20211217-113310-2', 'eval-EXP-20220103-175131', CLossV.D_LOSS_V4)
    arch_c.add_arch('M89', 'search-EXP-20211217-113322-2', 'eval-EXP-20220103-235822', CLossV.D_LOSS_V4)
    arch_c.add_arch('M90', 'search-EXP-20211217-113311-3', 'eval-EXP-20220104-021844', CLossV.D_LOSS_V4)
    arch_c.add_arch('M91', 'search-EXP-20211217-113326-3', 'eval-EXP-20220104-075957', CLossV.D_LOSS_V4)
    arch_c.save()


def genotype_correction():
    arch_c = ArchDataCollection()
    arch_c.load()
    arch_c.archs[
        'M2'].genotype_txt = "Genotype(normal=[('sep_conv_5x5', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 2), ('avg_pool_3x3', 0), ('max_pool_3x3', 3), ('max_pool_3x3', 0), ('dil_conv_3x3', 2)], reduce_concat=[2, 3, 4, 5])"
    arch_c.archs[
        'M3'].genotype_txt = "Genotype(normal=[('sep_conv_3x3', 0), ('dil_conv_5x5', 1), ('skip_connect', 2), ('skip_connect', 0), ('dil_conv_3x3', 2), ('dil_conv_3x3', 3), ('dil_conv_5x5', 4), ('sep_conv_3x3', 0)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('sep_conv_3x3', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('avg_pool_3x3', 3), ('dil_conv_3x3', 4), ('avg_pool_3x3', 2)], reduce_concat=[2, 3, 4, 5])"
    arch_c.archs[
        'M5'].genotype_txt = "Genotype(normal=[('dil_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('dil_conv_3x3', 0), ('avg_pool_3x3', 3), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('skip_connect', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('avg_pool_3x3', 2), ('skip_connect', 0), ('max_pool_3x3', 3), ('skip_connect', 0), ('avg_pool_3x3', 4), ('avg_pool_3x3', 1)], reduce_concat=range(2, 6))"
    arch_c.archs[
        'M9'].genotype_txt = "Genotype(normal=[('avg_pool_3x3', 1), ('dil_conv_5x5', 0), ('max_pool_3x3', 2), ('avg_pool_3x3', 0), ('dil_conv_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 4), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('avg_pool_3x3', 3), ('avg_pool_3x3', 1), ('avg_pool_3x3', 4), ('sep_conv_5x5', 3)], reduce_concat=[2, 3, 4, 5])"
    arch_c.save()


def add_closs_v():
    arch_c = ArchDataCollection()
    arch_c.load()
    # arch_c.archs['M1'].closs_v = CLossV.ORIGINAL
    arch_c.archs['M2'].closs_v = CLossV.ORIGINAL
    arch_c.archs['M3'].closs_v = CLossV.ORIGINAL
    arch_c.archs['M5'].closs_v = CLossV.ORIGINAL

    arch_c.archs['M6'].closs_v = CLossV.LEGACY
    arch_c.archs['M7'].closs_v = CLossV.LEGACY
    arch_c.archs['M8'].closs_v = CLossV.LEGACY
    arch_c.archs['M9'].closs_v = CLossV.LEGACY
    arch_c.archs['M10'].closs_v = CLossV.LEGACY
    arch_c.archs['M11'].closs_v = CLossV.LEGACY
    arch_c.archs['M12'].closs_v = CLossV.LEGACY
    arch_c.archs['M13'].closs_v = CLossV.LEGACY
    arch_c.archs['M14'].closs_v = CLossV.LEGACY

    arch_c.archs['M15'].closs_v = CLossV.ORIGINAL

    arch_c.archs['M16'].closs_v = CLossV.LEGACY

    arch_c.archs['M17'].closs_v = CLossV.ORIGINAL
    arch_c.archs['M19'].closs_v = CLossV.ORIGINAL

    arch_c.archs['M20'].closs_v = CLossV.LEGACY

    arch_c.archs['M21'].closs_v = CLossV.ORIGINAL

    arch_c.archs['M22'].closs_v = CLossV.LEGACY

    arch_c.archs['M23'].closs_v = CLossV.ORIGINAL

    arch_c.archs['M24'].closs_v = CLossV.LEGACY
    arch_c.archs['M25'].closs_v = CLossV.LEGACY
    arch_c.archs['M26'].closs_v = CLossV.LEGACY
    arch_c.archs['M27'].closs_v = CLossV.LEGACY
    arch_c.archs['M28'].closs_v = CLossV.LEGACY
    arch_c.archs['M29'].closs_v = CLossV.LEGACY
    arch_c.archs['M30'].closs_v = CLossV.LEGACY
    arch_c.archs['M31'].closs_v = CLossV.LEGACY
    arch_c.archs['M32'].closs_v = CLossV.LEGACY
    arch_c.archs['M33'].closs_v = CLossV.LEGACY
    arch_c.archs['M34'].closs_v = CLossV.LEGACY
    arch_c.archs['M35'].closs_v = CLossV.LEGACY
    arch_c.archs['M36'].closs_v = CLossV.LEGACY

    arch_c.archs['M37'].closs_v = CLossV.D_LOSS_V1
    arch_c.archs['M38'].closs_v = CLossV.D_LOSS_V1
    arch_c.archs['M39'].closs_v = CLossV.D_LOSS_V1
    arch_c.archs['M40'].closs_v = CLossV.D_LOSS_V1

    arch_c.archs['M41'].closs_v = CLossV.D_LOSS_V2
    arch_c.archs['M42'].closs_v = CLossV.D_LOSS_V2
    arch_c.archs['M43'].closs_v = CLossV.D_LOSS_V2
    arch_c.archs['M44'].closs_v = CLossV.D_LOSS_V2

    arch_c.archs['M45'].closs_v = CLossV.D_LOSS_V3
    arch_c.archs['M46'].closs_v = CLossV.D_LOSS_V3
    arch_c.archs['M47'].closs_v = CLossV.D_LOSS_V3
    arch_c.archs['M48'].closs_v = CLossV.D_LOSS_V3
    arch_c.archs['M49'].closs_v = CLossV.D_LOSS_V3
    arch_c.archs['M50'].closs_v = CLossV.D_LOSS_V3
    arch_c.archs['M51'].closs_v = CLossV.D_LOSS_V3
    arch_c.archs['M52'].closs_v = CLossV.D_LOSS_V3
    arch_c.archs['M53'].closs_v = CLossV.D_LOSS_V3
    arch_c.archs['M54'].closs_v = CLossV.D_LOSS_V3
    arch_c.archs['M55'].closs_v = CLossV.D_LOSS_V3
    arch_c.archs['M56'].closs_v = CLossV.D_LOSS_V4
    arch_c.archs['M57'].closs_v = CLossV.D_LOSS_V4
    arch_c.archs['M58'].closs_v = CLossV.D_LOSS_V4
    arch_c.archs['M59'].closs_v = CLossV.D_LOSS_V4
    arch_c.archs['M60'].closs_v = CLossV.D_LOSS_V4
    arch_c.archs['M61'].closs_v = CLossV.D_LOSS_V4
    arch_c.archs['M62'].closs_v = CLossV.D_LOSS_V4
    arch_c.archs['M63'].closs_v = CLossV.D_LOSS_V4
    arch_c.archs['M64'].closs_v = CLossV.D_LOSS_V4
    arch_c.archs['M65'].closs_v = CLossV.D_LOSS_V4
    arch_c.archs['M66'].closs_v = CLossV.D_LOSS_V4
    arch_c.archs['M67'].closs_v = CLossV.D_LOSS_V4
    arch_c.archs['M68'].closs_v = CLossV.D_LOSS_V4
    arch_c.archs['M69'].closs_v = CLossV.D_LOSS_V4
    arch_c.archs['M70'].closs_v = CLossV.D_LOSS_V4
    arch_c.archs['M71'].closs_v = CLossV.D_LOSS_V4

    arch_c.archs['M72'].closs_v = CLossV.ORIGINAL
    arch_c.archs['M73'].closs_v = CLossV.ORIGINAL
    arch_c.archs['M74'].closs_v = CLossV.ORIGINAL
    arch_c.archs['M75'].closs_v = CLossV.ORIGINAL
    arch_c.archs['M76'].closs_v = CLossV.ORIGINAL

    arch_c.save()
