# Пайплайн для обучения нейронок.

### Setup guide
"Начало работы" вот тут: https://confluence.vk.team/pages/viewpage.action?pageId=508183959

### Rules
* Все эксперименты хранятся в папке pipeline в соответсвующих кварталах. Новые эксперимены создаются только в текущем квартале.
* Если Вы обновляете старый эксперимент, то либо создаете копию в новом квартале (менее предпочтительно, делается только если предыдущий эксперимент тоже важно сохранить), либо просто переносим старый эксперимент в новый квартал и там уже меняем
* Если Вы обновляете torch/lightning в мастере, убедитесь, что все эксперименты в SOTA работают с актуальной версией

### Lightning/torch migration guide
Если требуется обновить старые эксперименты / что-то не работает, имеет смысл пройтись по чек-листу:
* Убедитесь, что внутри лайтнинг модуля нет prepare_data (надо заменить на setup)
def setup(self, stage)
* Убедитесь, что вы не используете trainer.proc_rank (переименовываем в global_rank)
self.trainer.global_rank
* trainig_step возырвщает чистый loss (а не в словаре), логирование выглядит вот так (обратите внимание на detach)
self.log('trainig_loss', loss.detach())
* Логирование метрик должно выглядеть вот так:
self.log('ndcg@5', ndcg, on_epoch=True, rank_zero_only=True)
self.log('auc', auc, on_epoch=True, rank_zero_only=True)
* Если во время обучения возникает ошибка "This error indicates that your module has parameters that were not used in producing loss"
from pytorch_lightning.plugins import DDPPlugin
И внутрь trainer:
plugins=[DDPPlugin(find_unused_parameters=True)]
* Если при запуске эксперимента внезапно возникает "Segmentation fault (core dumped)", попробуйте почистить кэш и затем, возможно, переустановить venv:
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

