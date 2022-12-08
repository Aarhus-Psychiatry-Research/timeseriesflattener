# Changelog

<!--next-version-placeholder-->

## v0.17.0 (2022-12-08)
### Feature
* More appropriate logging level ([`7122e79`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/7122e79606ec2d2fd5802361ecda8ebde318de12))
* Infer resolve_multiple_fn str rep for groups ([`8058d65`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/8058d65fdf6e65c534adab87861dcaaeafdd03b8))
* Infer resolve_multiple_fn str repr from __name__ ([`55be07d`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/55be07df7947b5a7baafafcae7955c4b426a45d0))
* Add lookahead and lookbehind days to feature group specs ([`318591b`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/318591b7edda233897e11af0a79fdd97a5f12716))

### Fix
* Guard against incident attribute not existing ([`3b1329b`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/3b1329b35c11477d09555287a904a9f895e99964))
* Re-add resolve_multiple str resolution ([`18983a7`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/18983a7988473352f2e994f1ef18ab7e1c8caa80))
* Only infer resolve_multiple_str if not specified manually ([`f2648f8`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/f2648f8afc692812f490e019db0bbb44391d96e2))
* Create dir for diskcache if it doesn't exist ([`0e32436`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/0e32436a0c9ee2d59b916158e7f46f0081661fa2))
* Create dir if it doesn't exist ([`c32f3c7`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/c32f3c702c0d473b59c506cd2e855def65764b24))

### Documentation
* Update output ([`a4fa6f7`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/a4fa6f7bf5d9f05d3b6c423f596a5bf102f06e49))
* Update tutorial based on feedback ([`92c3d3b`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/92c3d3bf5db019a2a3e5bd4ae3182f22cc4bcd38))
* Add figures to basic tutorial ([`5eb069f`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/5eb069f28bde3a4bb951b6d81f86ae80ed9b455f))
* Misc. updates to advanced notebook ([`a4a9380`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/a4a9380abffc76e55688744571049ff6b1588779))
* Groupspec add output type ([`0b3df30`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/0b3df30d9bc1287a596f746c5a7fcef58ab21b05))
* Update formatting in 02_advanced ([`47065e9`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/47065e90e2179fa464bf055a1c87ae8bfc8f3134))
* Initial stab at advanced tutorial ([`e8128bd`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/e8128bdfc3fd544b2d12a1bdf71b7de204187152))

## v0.16.0 (2022-12-07)
### Feature
* Add diff when dropping rows ([`0421ed8`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/0421ed800abf9c50fcd98dc3a6f50f72a740a7b2))
* No def arg for drop_if_insufficient_look_direction ([`d290153`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/d29015348bb90e5e063bf7ff1fcd5e15d75d2823))
* Drop pred times with insufficient look distance ([`8ec6e2c`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/8ec6e2c4e0fb5b97cad6bfac61164a4ca5fd54c6))
* First stab at dropping unused pred times ([`66bb7d4`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/66bb7d482fb484d7affa7401b537b246980700d5))
* Check that all specs have required columns in values_df ([`9da16f0`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/9da16f00cdd869dadca3b89fdab6c9694905450f))
* Better logging ([`44eb010`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/44eb0108daf00e400032ca47a0c635c310757e3c))
* Process all temporal specs in one batch ([`01b3957`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/01b3957aabc9b6fa99c9f306c5f5cefa76795cc2))
* When init spec, coerce timestamp if possible ([`4a6f817`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/4a6f81770ba897ed31ad1956d05422d5073ac9d2))
* Process specs all at once ([`4bcdb82`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/4bcdb82e46f39ccdf5d4a9503420f43cb5ebc79c))
* Collect specs with one interface ([`7135c4d`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/7135c4da1061af77ab595420b7d4935d71fc6582))

### Fix
* Incorrect dim comparison in diff decorator ([`c161f9a`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/c161f9ad06cde09d495065b8db8d8795554aa92b))
* Revert TemporalSpec renaming ([`3140cd5`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/3140cd52196f84553566ee0247ff8f8c0af5caf1))
* Missing column should be a keyerror ([`452e903`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/452e903ef389ff25b4cddbb1cfeafffbbfbe9cec))
* Undo renaming of PredictorSpec -> TemporalSpec ([`150035c`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/150035c01313a3bae8b21cd18dfc6e8ccd2ad12c))
* Don't process as batch if no specs to process ([`aba0b67`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/aba0b67928ffc0fcc64538cb2af7980b77d52b44))
* Only batch process predictors if any are added ([`c21737c`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/c21737cb1beff190946c5a3614c6ebeb225c059b))
* __eq__ in AnySpec ([`4c650d1`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/4c650d134c60c9cf13fe5bab1799b46f36823776))

### Documentation
* Improve from review suggestions ([`57804df`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/57804dfd9ba39a19351ed4efdbca2011cb37d24e))
* Improve from review suggestions ([`0316dbe`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/0316dbe67f02ed0faad0051818c2f08aadda7047))
* Elaborate on drop_pred_time ([`254fb3d`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/254fb3d438a4acacbc0176e0c5c15e6885bbffed))
* Update tutorial to new interface ([`bc0405e`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/bc0405e07b46c72e4691e144b741b39a3a7615fd))
* Raise valueerror if no prediction times remain ([`18c7f82`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/18c7f82095aed1acc37578541efa2b0a1e91ba88))
* Improve drop_records docs. ([`871ec18`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/871ec18e869c87bd78de5bc7b871111943c232a4))
* Minor docstring edit ([`1a7267a`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/1a7267a0d6c542943d03e955f47a6b67a1ea0106))

## v0.15.0 (2022-12-06)
### Feature
* Allow either interval_days or lookahead/lookbehind days ([`a270801`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/a2708014eaad78622a5447cf6958f544ad095945))

### Fix
* Failing imports after merge ([`dd17771`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/dd177711c8aff74619d962845ab7dd32df00a91b))
* Unify file naming in cache module ([`f155217`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/f155217edc860cbde68e60f6d102dc1fc2191347))
* Use correct suffix ([`6e737b8`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/6e737b8010c06f62ef320c868fc3821d524ec117))
* Remove seconds from diskcache to avoid ([`f9ac05c`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/f9ac05c46ca13649c248588470f000aa70ec6650))
* Key_for_resolve_multiple should be optional ([`c569b74`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/c569b749e9c47a60d0e66eae8c0633519ed0e0de))

### Documentation
* Add basic tutorial ([`8136a1b`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/8136a1b3db20f30df7898363dfa68e584a6cfe8d))

## v0.14.0 (2022-12-06)
### Feature
* Add colored logging ([`d230213`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/d230213b0fde3a26329e3c5bab6737ef3391fa09))
* Add logging by default ([`d254b69`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/d254b69057792a1088a1b35c70e2698206aedce4))
* Refactor flattened_dataset to use logging instead of msg ([`d9fc31d`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/d9fc31db5b4f42bb29cfed96a5edd2fbc1a43c7c))

### Documentation
* Improve API ([`44be982`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/44be9826772d4c167ef22e0cc856fb0812e67dd1))
* Add example of adding a root logger ([`64b0002`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/64b000266468230a1d1c0e0217f6bcff1aa9d557))

## v0.13.0 (2022-12-06)
### Feature
* Check that all col names exist in df before creating spec ([`7e75001`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/7e7500146c75b854f4a6565283a07364fd2c63e7))

### Documentation
* Style ([`d56926e`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/d56926e74b80c1bf777a5109b55ca3b711b4de82))

## v0.12.1 (2022-12-06)
### Fix
* Override cache attributes if unset or None ([`9f896c8`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/9f896c89be7f194c61de9d99ac0bfe8488ccb27f))
* Duplication of citizen id columns when reading cache ([`d78340c`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/d78340c25d4b8478b28b12104b9f0d9963bb59f3))

### Documentation
* Raise warning if overriding cache attributes ([`ffba27e`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/ffba27e05058892fe1a3c9f1e6d8c29d6cef66a6))
* Fixed citation ([`ef3312a`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/ef3312a90cc78318b0264ad86a1649ec2d502ba0))
* Update citation.cff with zonedo ([`ef3fc65`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/ef3fc6543dfc8807aefaa5d792f186580a8789f6))
* Added pypi badge ([`218cbcc`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/218cbcc241cf6b848f52dfde3a8acc410be057f0))
* Improvements ([`1726246`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/17262468476eb1899c536a267344dcaf1840568c))
* Misc. ([`338b2b1`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/338b2b11b3f46cc8d6fc308198462a938da2902f))

## v0.11.0 (2022-11-30)
### Documentation
* Added explanation comment to token ([`9ae8716`](https://github.com/Aarhus-Psychiatry-Research/timeseriesflattener/commit/9ae8716435b766eca0338b179446b569dbdf1ee6))

## **Notice**
`psycop-feature-generation` have been renamed timeseriesflattener

## v0.10.0 (2022-11-21)
### Feature
* Add n_hba1c_within_n_lookahead_days ([`e84b591`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/e84b5918724a55b721ec4d1a7291533227fe9ef8))
* Add outcome ([`cd39dd6`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/cd39dd6adfaa0c261abb2942ac9f215670c1c92d))
* Add birth year as a predictor ([`7b186d2`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/7b186d2fc339dd423207b9311cdb6d1fad7078ee))
* Allow exclusion of specific atc codes ([`75619a1`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/75619a122e26ad43fd7058e3db49c062e33b0b9f))

### Fix
* Date of birth col name should respect output prefix ([`6ec6535`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/6ec6535a2df4161ffc6e94e02eb9b340722f43e7))
* Incorrect column name when adding age as predictor ([`cdbf25c`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/cdbf25cd26f60baa795e43bc9df3865868248960))
* Errors in sql loaders after refactor ([`28c9f63`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/28c9f63fd8b81892fbea2695df94df47f6fe8dc6))
* Correct type hinting in load_diagnoses ([`f2d5c5b`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f2d5c5bfebce3fc8c3c61ee5231716dfc7883c8e))

### Documentation
* Speccify that n_rows = None returns all rows. ([`a4720a8`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a4720a8777601e81993f6707a4f4f48a6f850282))

### Performance
* Shuffle feature specs to even out compute vs. IO load ([`0db9f0f`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/0db9f0fd77989fdced4489ca9c45caff3d741086))
* Tweak n_workers for more performance ([`3eeee4d`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3eeee4da7092364d68a8a6eb2e3e028df4403fa1))
* Segment feature loading for more parallelisation ([`9ee5c87`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/9ee5c8778820da29d370653ce435665226e3cfdb))
* Rotate feature addition for debugging ([`76af9c7`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/76af9c717059f063d8aeb6756816b8e574bb845b))
* Parallelise temporal predictor loading ([`8d53f16`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/8d53f165e760e581d8888287474f6f353642ae0b))
* Only create one subprocess per values loader ([`1a3e5de`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/1a3e5dedb66a864b27be5318359b60f778eaa15b))
* Parralelise groupspec combination creation ([`9ccba2a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/9ccba2a24538b752409f166f82a0474805e18150))

## v0.9.0 (2022-11-18)
### Feature
* At groupspec init, iterate over values_loader and check that they exist in the loader registry ([`04dfd7e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/04dfd7e7e038472cfd26f67c79a6b050cc13b15e))

### Fix
* More explanation in error message ([`b784991`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/b7849911c85ca6ac5bd165b7a48ccce1a768f70b))
* Bettee valueerror message formatting ([`7b3b994`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/7b3b994cbe38df73a4149c4463b5f283ad297218))
* Better valueerror message ([`d92f798`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/d92f7989af27a879fd090bed33ce5027e96e581b))
* Find invalid loaders ([`ba2d4c5`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/ba2d4c540f097c33ca5c29a0b72a908ad6dc04e3))

## v0.8.0 (2022-11-17)
### Feature
* Allow load_medications to concat a list of medications ([`d78f465`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/d78f46592213b8245229d6618d40f1a1ff4d80eb))

### Fix
* Remove original functions ([`da59110`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/da59110978469b0743ce2d625005fc90950fb436))

### Documentation
* Improve docs ([`9aad0af`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/9aad0af6205af2e3deffb573676af5a20401bae1))

## v0.7.0 (2022-11-16)
### Feature
* Full run ([`142212f`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/142212fc63a59662048b6569dc874def92dfe62f))
* Rename resolve_multiple registry keys to their previous one ([`3fd3f35`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3fd3f3566a8a9312ef9a8326a700b162ed9815c3))
* Reimplement ([`c99585f`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/c99585fdf9f9f407a69e0ead05f935d34ed86a63))
* Use lru cache decorator for values_df loading ([`4006818`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/40068187da20854fcca980872bc42b8a3a096cc9))
* Add support for loader kwargs ([`127f821`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/127f8215c35b792390595b890210baa0e8cf3591))
* Move values_df resolution to anyspec object ([`714e83f`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/714e83fd3722b298cdd256b06915659ca7a34259))
* Make date of birth output prefix a param ([`0ed1198`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/0ed11982ba1b239e5650d23dbfab707100e38137))
* Ensure that dfs are sorted and of same length before concat ([`84a4d65`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/84a4d65b731a6822d0a8f6313d01b7de9c574afe))
* Use pandas with set_index for concat ([`b93290a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/b93290ae733857855abe8197291dd047cf6c6fa8))
* Use pandas with set_index for concat ([`995da41`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/995da419baef8fdb1f205610d63805c152156474))
* Speed up dask join by using index ([`3402281`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/34022814b6e9c93a715a2d6343f7c038feb6a932))
* Require feature name for all features, ensures proper specification ([`6af454a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/6af454a325bdb07a37c435246b0ead4d4dad971b))
* First stab at adapting generate_main ([`7243130`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/724313073d5eb225b3eddba597064f35053b0bd4))
* Add exclusion timestamp ([`b02de1a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/b02de1a92f12545bc1ac0ea40f98468f21185259))
* Improve dd.concat ([`429da34`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/429da346b0de1e07809176a1d2d34962c7e9770a))
* Handle strs for generate_feature_spec ([`7d54488`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/7d5448853ba3bdd0b13071afbb2c738d741337d3))
* Convert to dd before concat ([`06101d8`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/06101d86561af56eebaea2090baaf27aa3747b71))
* Add n hba1c ([`3780d84`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3780d841699d2a6b9077ca4fa3117d69f32bb123))
* Add n hba1c ([`614245e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/614245ead3fcc5b554a26ba515ff689d2627429b))

### Fix
* Coerce by default ([`60adb99`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/60adb999c83b6d93d97f1c6537f20c012721561e))
* Output_col_name_override applied at loading, not flattening ([`95a96ce`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/95a96ce64a186c01f4e4e09d8787a97e42388df8))
* Typo ([`01240ed`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/01240ed7b06843011593bcb3c3c71283918c90b2))
* Incorrect attribute addressing ([`a6e82b5`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a6e82b59ca353413066346e089f1557dc831d145))
* Correctly resolve values_df ([`def67cd`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/def67cd954440df76f1570acf7e48f68ae636d6c))
* MinGroupSpec should take a sequence of name to permute over ([`f0c8140`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f0c814017b6f355d5916ba15fe26d9f3350a3a7b))
* Typo ([`61c7241`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/61c7241d11f7bff3bad11e98cfea38600e239167))
* Remove resolve_multiple_fn_name ([`617d386`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/617d386095139bc3445a5f4d14ffebce1e5ffa24))
* Old concat resulted in wrong ordering of rrows. ([`3759f71`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3759f719070175c8be4184a0bdc5fc07db2c492c))
* Set hba1c as eval ([`89fe6d2`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/89fe6d209b93d345d9a0d8cd562e90ec395dfa8d))
* Typos ([`6eac440`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/6eac4408d8f0a58bb4cc66ac948bae5519a2c8cd))
* Correct col name inference for static predictors ([`dfe5dc7`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/dfe5dc72d5d22332ce3d496fb1d3bcca3c9328c7))
* Misc. fixes ([`45f8348`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/45f83488bef809ae059825caea9bf6937a5264d9))
* Generate the correct amount of combinations when creating specs ([`c472b3c`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/c472b3c69e0dfc64b433546e538298ddd2d44a5f))
* Typo resulted in cache breaking ([`fdd47d7`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/fdd47d705f166fcc3dc54612dc0387761d0489a9))
* Correct col naming ([`bc74ae3`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/bc74ae3089a7bbfc99ee31d82902e1c98e30f18e))
* Do not infer feature name from values_df ([`150569f`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/150569fde483f6c427f1efe5688038340dfceb92))
* Misc. errors found from tests ([`3a1b5db`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/3a1b5db493566592b349d317f7641d7564a662ad))
* Revert falttened dataset to use specs ([`e4fada7`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/e4fada7a9fb98d1ebccd6c41568619aa7e059d79))
* Misc. errors after introducing feature specs ([`0308eca`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/0308ecae8032ff309725b0917fd3901fadf102f9))
* Correctly merge dataframes ([`a907885`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a907885f592ba345cdf68ce5299699aacdc97b49))
* Cache error because of loss off UUID ([`89d7f6f`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/89d7f6f0ce557c7c3126116864ba75d0ddb0037e))
* New bugs in resolve_multiple ([`5714a39`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/5714a39c9e84081f6429dd0b8119873a9610e804))
* Rename outcomespec appropriately ([`41fa220`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/41fa22069453ac6df7dae824d49944775cf12ecc))
* Lookbehind_days must be iterable ([`cc879e9`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/cc879e9d6b0f806a2a604ff71cb3febbd625c2aa))

### Documentation
* Document feature spec objects ([`c7f1074`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/c7f10749d49b14a4614436097de2478f3e7fc879))
* Typo ([`6bc7140`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/6bc71405a318de4811f259b2823c91f1951ebb95))

### Performance
* Move pd->dd into subprocesses ([`dc5f38d`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/dc5f38db7d09900955e475d9c87837dab207ba9b))

## v0.6.3 (2022-10-18)
### Fix
* Remove shak_code + operator check ([`f97aee8`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f97aee8ff932270abed737308591cc87678062a8))

## v0.6.2 (2022-10-17)
### Fix
* Ignore cat_features ([`2052505`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/20525056d6e97aceb277a5e05cde3d8e701650e3))
* Failing test ([`f8190b4`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f8190b47b020782e1029f875bc3acee5c3abe566))
* Incorrect 'latest' and handling of NaN in cache ([`dc33f7e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/dc33f7ef68c065814779f44b7dd8e65c46755fea))

## v0.6.1 (2022-10-13)
### Fix
* Check for value column prediction_times_df ([`5356464`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/5356464ee5dbe302cf2bafd3203be88016e6bcaf))
* Change variable name ([`990a848`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/990a848a7d63410d06e491664d549f04a24a4384))
* More flex loaders ([`bcad700`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/bcad70092069cb818a67383bd8a925248edf04cd))

## v0.6.0 (2022-10-13)
### Feature
* Use wandb to monitor script errors ([`67ae9b9`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/67ae9b9ebecef68d4d0ceb74b58dc7bd3f6798b6))

### Fix
* Duplicate loading when pre_loading dfs ([`7f864dc`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/7f864dca9315b296e16cc1c9efd84e73627c9e2f))

## v0.5.2 (2022-10-12)
### Fix
* Change_per_day function ([`bf4f18c`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/bf4f18c10c66b8daa660d9ad9bb0dd05361dde75))
* Change_per_day function ([`b11bcaa`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/b11bcaaaac0e8de75e798491b0e4355220029773))

## v0.5.1 (2022-10-10)
### Fix
* Change_per_day functions ([`d696389`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/d6963894c458cdacc43cec579af1452a427ab86f))
* Change_per_day function ([`4c8c118`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/4c8c118e9f0e53c145ad07132afcc475890cb021))

## v0.5.0 (2022-10-10)
### Feature
* Add variance to resolve multiple functions ([`8c471df`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/8c471df351855a5f7b16734f999c73ae0e590874))

### Fix
* Add vairance resolve multiple ([`7a64c5b`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/7a64c5ba6d776cea6bf7b8064698bf9ad4d6814e))

## v0.4.4 (2022-10-10)
### Fix
* Deleted_irritating_blank_space ([`a4cdfc5`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/a4cdfc58ccf7524a308af1bab3b0ca6f0b15e834))

## v0.4.3 (2022-10-10)
### Fix
* Auto inferred cat features ([`ea0d946`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/ea0d946cbf658d8d7e22d45363f9dd7d5a7e3fff))
* Auto inferred cat features error ([`f244715`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/f2447156beef5128819f97f7a9554d03d394e01a))
* Resolves errors caused from auto cat features ([`667a905`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/667a9053f89413ada54624ae19d0d7e880724573))

## v0.4.2 (2022-10-06)
### Fix
* Incorrect function argument ([`33e0a3e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/33e0a3e959a2cf864c2494810741b02d073c55c4))
* Expanded test to include outcome, now passes locally ([`640e7ec`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/640e7ec9b0ed294db2e58ae56d1a06740b4e8855))
* Passing local tests ([`6ed4b2e`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/6ed4b2e03f42f257342ae62b11302d76449a1cdc))
* First stab at bug fix ([`339d793`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/339d7935c0870bbdd140547d9d3e63881f07a6e8))

## v0.4.1 (2022-10-06)
### Fix
* Add parents to wandb dir init ([`5eefe3a`](https://github.com/Aarhus-Psychiatry-Research/psycop-feature-generation/commit/5eefe3aa14dbe2cd3e8d422c0224f3eb557da0df))

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
