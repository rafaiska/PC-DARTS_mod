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
    D_LOSS_V5 = 8  # V4 but with a correct set of parameters for CustomLoss(nn.CrossEntropyLoss) (op_oracle.py)
    BOGUS_ORIGINAL = 7  # Original PC-DARTS, but using modified script. Unreliable as an original experiment


class HPCCluster(Enum):
    SDUMONT = 1
    CENAPAD = 2
    LMCAD = 3
    FINGANFORN = 4


def quotes(s):
    return '"' + s + '"'


class ArchData:
    ARCH_CSV_HEADER = ['arch_id', 'train_search_id', 'train_search_sys', 'best_train_id', 'best_train_sys',
                       'super_model_acc', 'model_acc', 'time_for_100_inf', 'macs_count', 'genotype_txt', 'closs_v',
                       'closs_w', 'fp_op_count']

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
        self.fp_op_count = None
        self.train_search_sys = None
        self.best_train_sys = None

    def __str__(self):
        return quotes('","'.join([str(self.__getattribute__(k)) for k in ArchData.ARCH_CSV_HEADER]))

    @staticmethod
    def get_csv_header():
        return quotes('","'.join(ArchData.ARCH_CSV_HEADER))


class ArchDataCollection:
    def __init__(self, collection_file_path='{}/.arch_data'.format(expanduser('~'))):
        self.collection_file_path = collection_file_path
        self.archs = None

    def select(self, closs_vs, train_search_sys=HPCCluster.SDUMONT, best_train_sys=HPCCluster.CENAPAD,
               closs_w_ht0=False, has_macs=True, has_acc=True):
        filtered = self.archs.values()
        filtered = list(filter(lambda a: a.closs_v in closs_vs, filtered))
        if train_search_sys:
            filtered = list(filter(lambda a: a.train_search_sys == train_search_sys, filtered))
        if best_train_sys:
            filtered = list(filter(lambda a: a.best_train_sys == best_train_sys, filtered))
        if has_acc:
            filtered = list(filter(lambda a: a.model_acc, filtered))
        if has_macs:
            filtered = list(filter(lambda a: a.macs_count, filtered))
        if closs_w_ht0:
            filtered = list(filter(lambda a: a.closs_w is None or a.closs_w > 0.0, filtered))
        return {a.arch_id: a for a in filtered}

    def save(self):
        with open(self.collection_file_path, 'wb') as fp:
            pickle.dump(self.archs, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        try:
            with open(self.collection_file_path, 'rb') as fp:
                self.archs = pickle.load(fp)
        except FileNotFoundError:
            self.archs = {}

    def add_arch(self, arch_id, train_search_id, best_train_id=None, closs_v=None,
                 train_search_sys=None, best_train_sys=None):
        if arch_id not in self.archs:
            a = ArchData()
            a.arch_id = arch_id
            a.train_search_id = train_search_id
            a.closs_v = closs_v
            a.train_search_sys = train_search_sys
            self.archs[arch_id] = a
        else:
            a = self.archs[arch_id]
        a.best_train_id = best_train_id
        if best_train_sys is not None:
            a.best_train_sys = best_train_sys

    def remove_arch(self, arch_id):
        if arch_id in self.archs:
            del self.archs[arch_id]
        else:
            raise RuntimeError

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
    arch_c.add_arch('M92', 'search-EXP-20220314-115704-0', 'eval-EXP-20220315-095026', CLossV.D_LOSS_V4)
    arch_c.add_arch('M93', 'search-EXP-20220314-115705-1', 'eval-EXP-20220315-224925', CLossV.D_LOSS_V4)
    arch_c.add_arch('M94', 'search-EXP-20220314-115704-2', 'eval-EXP-20220316-103156', CLossV.D_LOSS_V4)
    arch_c.add_arch('M95', 'search-EXP-20220314-115704-3', 'eval-EXP-20220316-181950', CLossV.D_LOSS_V4)
    arch_c.add_arch('M96', 'search-EXP-20220319-185257-0', 'eval-EXP-20220320-110447', CLossV.ORIGINAL)
    arch_c.add_arch('M97', 'search-EXP-20220319-185244-0', 'eval-EXP-20220321-031315', CLossV.ORIGINAL)
    arch_c.add_arch('M98', 'search-EXP-20220319-185258-1', 'eval-EXP-20220321-194236', CLossV.ORIGINAL)
    arch_c.add_arch('M99', 'search-EXP-20220319-185244-1', 'eval-EXP-20220322-101544', CLossV.ORIGINAL)
    arch_c.add_arch('M100', 'search-EXP-20220319-185258-2', 'eval-EXP-20220322-221733', CLossV.ORIGINAL)
    arch_c.add_arch('M101', 'search-EXP-20220319-185244-2', 'eval-EXP-20220323-085108', CLossV.ORIGINAL)
    arch_c.add_arch('M102', 'search-EXP-20220319-185257-3', 'eval-EXP-20220323-172146', CLossV.ORIGINAL)
    arch_c.add_arch('M103', 'search-EXP-20220319-185244-3', 'eval-EXP-20220324-024848', CLossV.ORIGINAL)
    arch_c.add_arch('M104', 'search-EXP-20220320-103237-0', 'eval-EXP-20220324-173027', CLossV.ORIGINAL)
    arch_c.add_arch('M105', 'search-EXP-20220320-103241-3', 'eval-EXP-20220325-063558', CLossV.ORIGINAL)
    arch_c.add_arch('M106', 'search-EXP-20220320-103242-1', 'eval-EXP-20220325-175132', CLossV.ORIGINAL)
    arch_c.add_arch('M107', 'search-EXP-20220320-103243-2', 'eval-EXP-20220326-083532', CLossV.ORIGINAL)
    arch_c.add_arch('M108', 'search-EXP-20220320-103251-0', 'eval-EXP-20220328-103407', CLossV.ORIGINAL)
    arch_c.add_arch('M109', 'search-EXP-20220320-103257-3', 'eval-EXP-20220328-221049', CLossV.ORIGINAL)
    arch_c.add_arch('M110', 'search-EXP-20220320-103258-1', 'eval-EXP-20220329-115904', CLossV.ORIGINAL)
    arch_c.add_arch('M111', 'search-EXP-20220320-103259-2', 'eval-EXP-20220330-052928', CLossV.ORIGINAL)
    arch_c.add_arch('M112', 'search-EXP-20220401-230817-2', 'eval-EXP-20220402-082907', CLossV.ORIGINAL)
    arch_c.add_arch('M113', 'search-EXP-20220401-230819-1', 'eval-EXP-20220403-024942', CLossV.ORIGINAL)
    arch_c.add_arch('M114', 'search-EXP-20220401-230819-3', 'eval-EXP-20220403-025651', CLossV.ORIGINAL)
    arch_c.add_arch('M115', 'search-EXP-20220401-230823-0', 'eval-EXP-20220403-215247', CLossV.ORIGINAL)
    arch_c.add_arch('M116', 'search-EXP-20220401-230835-2', 'eval-EXP-20220404-021504', CLossV.ORIGINAL)
    arch_c.add_arch('M117', 'search-EXP-20220401-230837-1', 'eval-EXP-20220404-160823', CLossV.ORIGINAL)
    arch_c.add_arch('M118', 'search-EXP-20220401-230837-3', 'eval-EXP-20220404-194751', CLossV.ORIGINAL)
    arch_c.add_arch('M119', 'search-EXP-20220401-230840-0', 'eval-EXP-20220405-090555', CLossV.ORIGINAL)
    arch_c.add_arch('M120', 'search-EXP-20220402-081719-0', 'eval-EXP-20220405-092938', CLossV.ORIGINAL)
    arch_c.add_arch('M121', 'search-EXP-20220402-081720-1', 'eval-EXP-20220405-214306', CLossV.ORIGINAL)
    arch_c.add_arch('M122', 'search-EXP-20220402-081720-3', 'eval-EXP-20220406-021242', CLossV.ORIGINAL)
    arch_c.add_arch('M123', 'search-EXP-20220402-081721-2', 'eval-EXP-20220406-124001', CLossV.ORIGINAL)
    arch_c.add_arch('M124', 'search-EXP-20220402-081731-0', 'eval-EXP-20220406-162747', CLossV.ORIGINAL)
    arch_c.add_arch('M125', 'search-EXP-20220402-081738-1', 'eval-EXP-20220407-014419', CLossV.ORIGINAL)
    arch_c.add_arch('M126', 'search-EXP-20220402-081738-3', 'eval-EXP-20220407-073829', CLossV.ORIGINAL)
    arch_c.add_arch('M127', 'search-EXP-20220402-081739-2', 'eval-EXP-20220407-141117', CLossV.ORIGINAL)
    arch_c.add_arch('M128', 'search-EXP-20220404-133031-0', 'eval-EXP-20220408-015421', CLossV.D_LOSS_V4)
    arch_c.add_arch('M129', 'search-EXP-20220404-133046-0', 'eval-EXP-20220408-054146', CLossV.D_LOSS_V4)
    arch_c.add_arch('M130', 'search-EXP-20220404-133026-1', 'eval-EXP-20220415-023440', CLossV.D_LOSS_V4)
    arch_c.add_arch('M131', 'search-EXP-20220404-133042-1', 'eval-EXP-20220409-032454', CLossV.D_LOSS_V4)
    arch_c.add_arch('M132', 'search-EXP-20220404-133022-2', 'eval-EXP-20220411-075434', CLossV.D_LOSS_V4)
    arch_c.add_arch('M133', 'search-EXP-20220404-133034-2', 'eval-EXP-20220412-051145', CLossV.D_LOSS_V4)
    arch_c.add_arch('M134', 'search-EXP-20220404-133022-3', 'eval-EXP-20220412-091218', CLossV.D_LOSS_V4)
    arch_c.add_arch('M135', 'search-EXP-20220404-133033-3', 'eval-EXP-20220412-222854', CLossV.D_LOSS_V4)
    arch_c.add_arch('M136', 'search-EXP-20220412-155259-0', 'eval-EXP-20220413-082912', CLossV.D_LOSS_V5)
    arch_c.add_arch('M137', 'search-EXP-20220412-155313-0', 'eval-EXP-20220413-152141', CLossV.D_LOSS_V5)
    arch_c.add_arch('M138', 'search-EXP-20220412-155300-1', 'eval-EXP-20220413-180005', CLossV.D_LOSS_V5)
    arch_c.add_arch('M139', 'search-EXP-20220412-155313-1', 'eval-EXP-20220413-210017', CLossV.D_LOSS_V5)
    arch_c.add_arch('M140', 'search-EXP-20220412-155300-2', 'eval-EXP-20220414-004037', CLossV.D_LOSS_V5)
    arch_c.add_arch('M141', 'search-EXP-20220412-155312-2', 'eval-EXP-20220414-033101', CLossV.D_LOSS_V5)
    arch_c.add_arch('M142', 'search-EXP-20220412-155300-3', 'eval-EXP-20220414-101103', CLossV.D_LOSS_V5)
    arch_c.add_arch('M143', 'search-EXP-20220412-155311-3', 'eval-EXP-20220414-200057', CLossV.D_LOSS_V5)
    arch_c.add_arch('M144', 'search-EXP-20220416-142626-0', 'eval-EXP-20220421-225903', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M145', 'search-EXP-20220416-225000-0', 'eval-EXP-20220421-201218', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M146', 'search-EXP-20220416-142641-0', 'eval-EXP-20220421-133614', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M147', 'search-EXP-20220416-224943-0', 'eval-EXP-20220421-031644', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M148', 'search-EXP-20220416-142639-1', 'eval-EXP-20220421-011326', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M149', 'search-EXP-20220416-142626-1', 'eval-EXP-20220420-143601', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M150', 'search-EXP-20220416-225000-1', 'eval-EXP-20220420-104017', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M151', 'search-EXP-20220416-224943-1', 'eval-EXP-20220420-011140', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M152', 'search-EXP-20220416-225001-2', 'eval-EXP-20220419-234458', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M153', 'search-EXP-20220416-142626-2', 'eval-EXP-20220419-125745', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M154', 'search-EXP-20220416-224943-2', 'eval-EXP-20220419-051644', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M155', 'search-EXP-20220416-142640-2', 'eval-EXP-20220419-021308', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M156', 'search-EXP-20220416-142626-3', 'eval-EXP-20220418-190245', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M157', 'search-EXP-20220416-225000-3', 'eval-EXP-20220418-185548', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M158', 'search-EXP-20220416-142641-3', 'eval-EXP-20220418-093535', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M159', 'search-EXP-20220416-224943-3', 'eval-EXP-20220418-093527', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M160', 'search-EXP-20220418-231106-0', 'eval-EXP-20220424-095257', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M161', 'search-EXP-20220418-231123-0', 'eval-EXP-20220424-021615', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M162', 'search-EXP-20220418-231106-1', 'eval-EXP-20220423-203035', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M163', 'search-EXP-20220418-231123-1', 'eval-EXP-20220423-105424', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M164', 'search-EXP-20220418-231107-2', 'eval-EXP-20220423-033202', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M165', 'search-EXP-20220418-231124-2', 'eval-EXP-20220422-230046', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M166', 'search-EXP-20220418-231113-3', 'eval-EXP-20220422-133246', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M167', 'search-EXP-20220418-231131-3', 'eval-EXP-20220422-085626', CLossV.ORIGINAL,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M168', 'search-EXP-20220510-121453-0', 'eval-EXP-20220512-154646', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M169', 'search-EXP-20220510-121452-0', 'eval-EXP-20220512-154701', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M170', 'search-EXP-20220510-121505-0', 'eval-EXP-20220512-194147', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M171', 'search-EXP-20220510-121507-0', 'eval-EXP-20220512-221653', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M172', 'search-EXP-20220510-121452-1', 'eval-EXP-20220513-022737', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M173', 'search-EXP-20220510-121505-1', 'eval-EXP-20220513-050153', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M174', 'search-EXP-20220510-121504-1', 'eval-EXP-20220513-062937', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M175', 'search-EXP-20220510-121452-2', 'eval-EXP-20220513-094104', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M176', 'search-EXP-20220510-121505-2', 'eval-EXP-20220513-114251', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M177', 'search-EXP-20220510-121504-2', 'eval-EXP-20220513-134114', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M178', 'search-EXP-20220510-121452-3', 'eval-EXP-20220513-171451', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M179', 'search-EXP-20220510-121507-3', 'eval-EXP-20220513-174538', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M180', 'search-EXP-20220510-121506-3', 'eval-EXP-20220513-211852', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M181', 'search-EXP-20220510-121454-3', 'eval-EXP-20220514-013020', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M182', 'search-EXP-20220512-140214-0', 'eval-EXP-20220514-030448', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M183', 'search-EXP-20220512-140231-0', 'eval-EXP-20220514-081416', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M184', 'search-EXP-20220512-140216-0', 'eval-EXP-20220514-095943', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M185', 'search-EXP-20220512-140227-0', 'eval-EXP-20220514-200018', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M186', 'search-EXP-20220512-140228-1', 'eval-EXP-20220515-013326', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M187', 'search-EXP-20220512-140215-1', 'eval-EXP-20220515-130336', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M188', 'search-EXP-20220512-140214-1', 'eval-EXP-20220515-212533', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M189', 'search-EXP-20220512-140230-1', 'eval-EXP-20220516-000351', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M190', 'search-EXP-20220512-140214-2', 'eval-EXP-20220516-053346', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M191', 'search-EXP-20220512-140227-2', 'eval-EXP-20220516-144759', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M192', 'search-EXP-20220512-140217-3', 'eval-EXP-20220516-195441', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M193', 'search-EXP-20220512-140232-3', 'eval-EXP-20220518-010215', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M194', 'search-EXP-20220512-140231-3', 'eval-EXP-20220518-050621', CLossV.D_LOSS_V5,
                    HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M195', 'search-EXP-20220512-231529-0', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M196', 'search-EXP-20220512-231531-0', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M197', 'search-EXP-20220512-231529-1', 'eval-EXP-20220524-231015', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M198', 'search-EXP-20220512-231529-2', 'eval-EXP-20220528-074608', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M199', 'search-EXP-20220512-231542-2', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M200', 'search-EXP-20220512-231529-3', 'eval-EXP-20220525-163149', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M201', 'search-EXP-20220512-231543-3', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M202', 'search-EXP-20220513-085517-2', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M203', 'search-EXP-20220513-085518-0', 'eval-EXP-20220519-190734', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M204', 'search-EXP-20220513-085530-1', 'eval-EXP-20220519-191017', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M205', 'search-EXP-20220513-085531-0', 'eval-EXP-20220520-020219', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M206', 'search-EXP-20220513-103031-1', 'eval-EXP-20220520-072610', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M207', 'search-EXP-20220513-103031-2', 'eval-EXP-20220520-212840', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M208', 'search-EXP-20220513-103046-1', 'eval-EXP-20220521-032432', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M209', 'search-EXP-20220513-174659-1', 'eval-EXP-20220522-081104', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M210', 'search-EXP-20220513-174700-0', 'eval-EXP-20220523-060221', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M211', 'search-EXP-20220513-174715-1', 'eval-EXP-20220524-001330', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M212', 'search-EXP-20220513-174716-0', 'eval-EXP-20220525-210758', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M213', 'search-EXP-20220515-234914-0', 'eval-EXP-20220529-055527', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M214', 'search-EXP-20220515-234915-1', 'eval-EXP-20220529-074932', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M215', 'search-EXP-20220515-234930-0', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M216', 'search-EXP-20220515-234931-1', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M217', 'search-EXP-20220515-234930-2', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M218', 'search-EXP-20220515-234914-2', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M219', 'search-EXP-20220513-174700-2', 'eval-EXP-20220524-013050', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M220', 'search-EXP-20220513-103047-2', 'eval-EXP-20220522-103156', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M221', 'search-EXP-20220513-174716-2', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M222', 'search-EXP-20220513-103046-3', 'eval-EXP-20220521-131308', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M223', 'search-EXP-20220513-085531-3', 'eval-EXP-20220520-122415', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M224', 'search-EXP-20220515-234910-3', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M225', 'search-EXP-20220513-174654-3', 'eval-EXP-20220523-045450', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M226', 'search-EXP-20220517-214827-0', 'eval-EXP-20220527-092059', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M227', 'search-EXP-20220517-214816-0', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M228', 'search-EXP-20220517-214828-0', 'eval-EXP-20220527-092101', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M229', 'search-EXP-20220517-214818-1', 'eval-EXP-20220528-072203', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M230', 'search-EXP-20220517-214817-1', 'eval-EXP-20220521-173250', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M231', 'search-EXP-20220517-214832-1', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M232', 'search-EXP-20220517-214833-1', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M233', 'search-EXP-20220517-214816-2', 'eval-EXP-20220520-030757', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M234', 'search-EXP-20220517-214817-2', 'eval-EXP-20220524-202150', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M235', 'search-EXP-20220517-214832-2', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M236', 'search-EXP-20220517-214830-2', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M237', 'search-EXP-20220517-214816-3', 'eval-EXP-20220520-172227', CLossV.D_LOSS_V5, HPCCluster.SDUMONT, HPCCluster.CENAPAD)
    arch_c.add_arch('M238', 'search-EXP-20220517-214830-3', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M239', 'search-EXP-20220517-214829-3', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)
    arch_c.add_arch('M240', 'search-EXP-20220518-122508-0', '', CLossV.D_LOSS_V5, HPCCluster.SDUMONT)

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


def dump_archs_csv():
    csv_path = '{}/archs.csv'.format(expanduser('~'))
    arch_c = ArchDataCollection()
    arch_c.load()
    arch_c.csv_dump(csv_path)


def set_train_search_sys():
    arch_c = ArchDataCollection()
    arch_c.load()
    for a in arch_c.archs.values():
        a.train_search_sys = HPCCluster.SDUMONT
    arch_c.archs['M2'].train_search_sys = HPCCluster.LMCAD
    arch_c.archs['M5'].train_search_sys = HPCCluster.CENAPAD
    arch_c.archs['M6'].train_search_sys = HPCCluster.LMCAD
    arch_c.archs['M7'].train_search_sys = HPCCluster.LMCAD
    arch_c.save()


def set_best_train_sys():
    arch_c = ArchDataCollection()
    arch_c.load()
    cenapad_archs = set()
    with open('/home/rafael/cenapad_archs.txt', 'r') as fp:
        for line in fp:
            cenapad_archs.add(line[:-1])
    for a in arch_c.archs.values():
        if a.arch_id in cenapad_archs:
            a.best_train_sys = HPCCluster.CENAPAD
        else:
            a.best_train_sys = HPCCluster.SDUMONT
    arch_c.archs['M2'].best_train_sys = HPCCluster.LMCAD
    arch_c.save()
