# Changelog

<!--next-version-placeholder-->

## v0.4.0 (2022-10-06)
### Feature
* Add BMI loader ([`b6681ea`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/b6681ea3dc9f0b366666fb4adb964d453c094844))

### Fix
* Refactor feature spec generation ([`17e9f16`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/17e9f166aa48b2ed86f4490ac97a606232e8aeaa))
* Align arguments with colnames in SQL ([`09ae5f7`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/09ae5f7b91523c53431e6ef52f3ec6b382b70224))
* Refactor feature specification ([`373b0f0`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/373b0f025d4d74bc0041c3caa2ef8cf7559888ff))

## v0.3.2 (2022-10-05)
### Fix
* Hardcoded file suffix ([`0101acc`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/0101accb995d060908b28f1338a313d82661683a))

## v0.3.1 (2022-10-05)
### Fix
* Mismatched version in .tomll ([`292979b`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/292979bf85401818d5837a159c30c88c67ac454d))

## v0.3.0 (2022-10-05)
### Feature
* Update PR template ([`dfbf153`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/dfbf153348594b8b0eaac0974fff7c69680c473d))
* Migrate to parquet ([`a027549`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/a027549cd1bc17527c8c28726748b724b639d510))
* Set ranges for dependencies ([`e98b2a7`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/e98b2a708356b167102fcf3f77bf1f623f34bf07))

### Fix
* Pass value_col only when necessary ([`dc1019f`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/dc1019f6f42510ea9482c1ad83790908b839ed15))
* Pass value_col ([`4674e4a`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/4674e4aef272469a1b68baab6656fba7d5b6b046))
* Don't remove NaNs, might be informative. ([`1ad5d81`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/1ad5d810cc7ea969ce190e13b7b4cb25be15de01))
* Remove parquet default argument except in top level functions ([`ec3a98b`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/ec3a98bca22bf8385a527cefd2c80dd80b3a60ff))
* Align .toml and release version ([`80adbde`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/80adbdeec8cde7b8c0b5e37393f2b48844c53639))
* Failing tests ([`b5e4321`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/b5e43215943777ffa5ac9d63f878b0a2358485cd))
* Incorrect feature sets path, linting ([`605ccb7`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/605ccb7c5a3cfb103efcda8f965e8a72ae52ae7f))
* Handle dicts for duplicate checking ([`34524c0`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/34524c055f1335ae703fbfce11f234c065c4ccb9))
* Check for duplicates in feature combinations ([`63ad162`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/63ad1628f750abdd58c24d9b6ea53a9be8ef6032))
* Remove duplicate alat key which prevented file saving ([`f0c3e00`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/f0c3e006c84cd41054fdbca4cf1266d9f393a059))
* Incorrect argumetn ([`b97d54b`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/b97d54b097986f452ae2f00f5bba2a6f051c1132))
* Linting ([`7406288`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/7406288d50ecfe9436f95726a6fd72c886478923))
* Use suffix instead of string parsing ([`cfa96f0`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/cfa96f0d768c1fbbeca372f93ab970535479f003))
* Refactor dataset loading into a separate function ([`bca8cbf`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/bca8cbfb861aecc995e657285a0ad4011b47e407))
* More migration to parquet ([`f1bc2b7`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/f1bc2b7f872ed17c28501acdb377cf385bbe9118))
* Mark hf embedding test as slow, only run if passing --runslow to pytest ([`0e03395`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/0e03395958f30d0aff400d7eb1f227808f57226c))

## v0.2.4 (2022-10-04)
### Fix
* Wandb not logging on overtaci. ([`3baab57`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/3baab57c7ac760a0056aefb95918501d4f03c17a))

## v0.2.3 (2022-10-04)
### Fix
* Use dask for concatenation, increases perf ([`4235f5c`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/4235f5c08958ac68f5d589e3c517017185461afa))

## v0.2.2 (2022-10-03)
### Fix
* Use pypi release of psycopmlutils ([`5283b05`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/5283b058bc67ac4a4142aaaa9a95a06f5418ef01))

## v0.2.1 (2022-10-03)
### Fix
* First release to pypi ([`c29aa3c`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/c29aa3c847bcdafbc8e60ff61b6c2218ab8c1356))

## v0.2.0 (2022-09-30)
### Feature
* Add test for chunking logic ([`199ee6b`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/199ee6ba62cd915b3885ad5101286d6caca7a72f))

### Fix
* Pre-commit edits ([`94af649`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/94af64938a1ba082a545141ed5d332dbdd1df867))
* Remove unnecessary comment ([`3931395`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/393139512dd58ebeec143499317425ca63b25e45))

## v0.1.0 (2022-09-30)
### Feature
* First release! ([`95a557c`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/95a557c50107b34bd3862f6fea69db7a7d3b8a33))
* Add automatic release ([`a5023e5`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a5023e571da1cbf29b11b7f82b7dbb3d93bff568))
* Update dependencies ([`34efeaf`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/34efeaf295b468c3ebd13b917e37b319df18ccf6))
* First rename ([`879bde9`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/879bde97033e627269f3ffe856035dfbe1e1ffb7))
* Init commit ([`cdcab07`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/cdcab074310c843a7e1b737d655136e95b1c62ed))

### Fix
* Force dtype for windows ([`2e6e8bf`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/2e6e8bf148db256f6a047354a474705c25af3156))
* Linting ([`5cdfcfa`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/5cdfcfa75a866919364bd5bbf264db4fcaa8fdda))
* Pre code-split import statements need to be updated ([`a9e0639`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a9e06390aba1fa5cdcb7d0e9918bc158dbdcaf26))
* Misspecified python version in action ([`fdde2d2`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/fdde2d2e2bc7f115a313809789833bcd8c845d6d))
