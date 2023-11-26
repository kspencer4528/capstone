DROP TABLE
    IF EXISTS shpretention_member_population;
CREATE TEMP TABLE shpretention_member_population AS
WITH zip_to_fip AS         
    (SELECT DISTINCT 
         ctd.county_key,
         LEFT(mhd.physical_zip_code, 5) AS zip_code,
         COUNT(DISTINCT mhd.member_key) AS zip_cnt,
         ROW_NUMBER() OVER (PARTITION BY zip_code 
                                ORDER BY zip_cnt DESC, county_key) rn
     FROM dw.member.member_history_dim mhd
     LEFT JOIN dw.reference.county_dim ctd
         ON mhd.physical_county_key = ctd.county_key
     WHERE 
         LEN(mhd.physical_zip_code) >= 5 -- exclude incomplete zip codes
         AND ctd.county_key <> 0 -- exclude missing county_key
         AND CAST(mhd.update_date as DATE) < ${window_stop}$
     GROUP BY 
         ctd.county_key,
         zip_code),
         
cte_population AS
    (SELECT DISTINCT
        --Identifiers
        mem.shp_member_id,
        mmf.member_month_key,
        mmf.member_key,
        mmf.member_month_year_month, 
        --Product information, group information are identical to the product for MA, and employer infos are none
        prd.product_type_code,
        prd.product_type_name,
        prd.product_name,
        elg.from_date,
        elg.thru_date,
        --Demographics
        agd.age_years AS age,
        agd.cms_age_range_code, 
        UPPER((agd.cms_age_range_desc)::text) AS cms_age_range_desc, -- cms has wider ranges
        agd.gender_code,
        mhd.physical_longitude AS member_physical_longitude, 
        mhd.physical_latitude AS member_physical_latitude, 
        LEFT(mhd.physical_zip_code, 5) AS member_physical_zip_code,
        mhd.physical_county_name as county_name,
        mhd.physical_postal_state_code as state_code,
        svi.rpl_theme1, -- Socioeconomic 
        svi.rpl_theme2, -- Household Composition & Disability 
        svi.rpl_theme3, -- Minority Status & Language
        svi.rpl_theme4, -- Housing Type & Transportation
        svi.rpl_themes, -- overall tract summary ranking  
        -- Flags
        mhd.esrd_flag AS member_esrd_flag, --it would only be filled in for Medicare or maybe all government products and not everyone
        edd.medical_coverage_flag AS part_c_flag,
        edd.pharmacy_coverage_flag AS part_d_flag 

    FROM dw.eligibility.member_month_fact mmf
    JOIN dw.member.member_dim mem 
        ON mem.member_key = mmf.member_key
    JOIN dw.member.member_history_dim mhd 
        ON mhd.member_history_key = mmf.member_history_key 
    JOIN dw.eligibility.product_dim prd 
        ON prd.product_key = mmf.product_key
    JOIN eligibility.eligibility_fact elg
        ON elg.eligibility_key = mmf.eligibility_key
    JOIN dw.eligibility.eligibility_detail_dim edd 
        ON edd.eligibility_detail_key = elg.eligibility_detail_key
    JOIN dw.reference.age_gender_dim agd 
        ON agd.age_gender_key = mmf.age_gender_key 
    LEFT JOIN zip_to_fip ztf
        ON LEFT(mhd.physical_zip_code, 5) = ztf.zip_code 
       AND rn = 1
    LEFT JOIN dw.reference.county_dim ctd
       ON NVL(NULLIF(mhd.physical_county_key, 0), ztf.county_key) = ctd.county_key 
    LEFT JOIN dw.sandbox.vw_datascience_svi svi
        ON ctd.fips_county_code = svi.fips
       AND mmf.member_month_date BETWEEN svi.from_date AND svi.thru_date 

    WHERE  
        edd.void_flag = 'N'
        AND edd.deleted_flag = 'N'
        AND mmf.member_month_status_desc in ('Active')
        AND ${window_stop}$ BETWEEN mhd.from_date and mhd.thru_date
        AND mem.date_of_death IS NULL
        AND mmf.member_month_date = TRUNC(DATE_TRUNC('month',DATE(${window_stop}$)))
        AND mmf.primacy_order = 1 -- primacy_order in eligibility.member_month_fact and eligibility.eligibility_fact can differ
        AND elg.primacy_order = 1 -- limit to members with SHP as primary enrollment
        AND prd.product_type_code in (7, 8, 9)
        AND agd.age_years > 18 -- I remember it was mentioned in one of our discussion
    )

SELECT 
    pop.shp_member_id,
    pop.member_month_key,
    pop.member_key,
    pop.member_month_year_month AS most_recent_member_year_month, 
    pop.product_type_code,
    pop.product_type_name,
    pop.product_name,
    pop.from_date,
    pop.thru_date,
    pop.age,
    pop.cms_age_range_code, 
    pop.cms_age_range_desc,
    pop.gender_code,
    pop.member_physical_longitude, 
    pop.member_physical_latitude, 
    pop.member_physical_zip_code,
    pop.county_name,
    pop.state_code,
    pop.rpl_theme1,
    pop.rpl_theme2, 
    pop.rpl_theme3,
    pop.rpl_theme4,
    pop.rpl_themes,
    pop.member_esrd_flag, 
    pop.part_c_flag,
    pop.part_d_flag 

FROM cte_population pop
;  

DROP TABLE  IF EXISTS shpretention_member_nextyr_population;
CREATE TEMP TABLE shpretention_member_nextyr_population AS
SELECT DISTINCT
        --Identifiers
        mem.shp_member_id,
        mmf.member_key,
        elg.from_date as ny_from_date,
        elg.thru_date as ny_thru_date

FROM dw.eligibility.member_month_fact mmf
JOIN dw.member.member_dim mem 
    ON mem.member_key = mmf.member_key
JOIN shpretention_member_population pop
    ON pop.member_key = mmf.member_key    
JOIN dw.eligibility.product_dim prd 
    ON prd.product_key = mmf.product_key
JOIN eligibility.eligibility_fact elg
    ON elg.eligibility_key = mmf.eligibility_key
JOIN dw.eligibility.eligibility_detail_dim edd 
    ON edd.eligibility_detail_key = elg.eligibility_detail_key
JOIN dw.reference.age_gender_dim agd 
    ON agd.age_gender_key = mmf.age_gender_key 

WHERE  
    edd.void_flag = 'N'
    AND edd.deleted_flag = 'N'
    AND mmf.member_month_status_desc in ('Active')
    AND mem.date_of_death IS NULL
    AND mmf.member_month_date = TRUNC(DATE_TRUNC('month',DATE(DATEADD(year, 1, ${window_stop}$))))
    AND mmf.primacy_order = 1 -- primacy_order in eligibility.member_month_fact and eligibility.eligibility_fact can differ
    AND elg.primacy_order = 1 -- limit to members with SHP as primary enrollment
    AND prd.product_type_code in (7, 8, 9)
    AND agd.age_years > 18 -- I remember it was mentioned in one of our discussion
;

DROP TABLE  IF EXISTS shpretention_quit_plan;
CREATE TEMP TABLE shpretention_quit_plan AS
    SELECT  pop.shp_member_id,
            CASE WHEN npop.shp_member_id IS NULL THEN 1    
            ELSE 0
            END AS quit_plan
     FROM  shpretention_member_population AS pop
     LEFT JOIN  shpretention_member_nextyr_population AS npop
            ON  npop.shp_member_id = pop.shp_member_id
;

DROP TABLE
    IF EXISTS shpretention_member_cchg_mara;
CREATE TEMP TABLE shpretention_member_cchg_mara AS     
SELECT
    pop.member_key,
    pop.member_month_key,
    CASE WHEN primary_cchg.cchg_grouping_code = 1 -- Healthy
        THEN 0
        ELSE cif.cchg_category_cnt 
    END AS cc_disease_cnt, -- Distinct count of chronic condition: The number of CCHG categories associated with the CCHG grouping
    primary_cchg.cchg_grouping_code AS cchg_cat_rollup_code, -- 2=Chronic Condition, 1=Healthy, -1=No CCHG Grouping
    primary_cchg.cchg_grouping_desc AS cchg_cat_rollup_desc, 
    CASE WHEN primary_cchg.cchg_grouping_code = 1 -- Healthy
        THEN 999 -- a number not reserved by cchg
        ELSE primary_cchg.cchg_category_code 
    END AS cchg_cat,
    primary_cchg.cchg_category_desc AS cchg_desc,
    cdd.cancer_severity_code, 
    cdd.cancer_severity_desc, 
    cdd.cancer_active_code, 
    cdd.cancer_active_desc, 
    cdd.copd_severity_code, 
    cdd.copd_severity_desc, 
    cdd.diabetes_severity_code, 
    cdd.diabetes_severity_desc, 
    cdd.hypertension_severity_code, 
    cdd.hypertension_severity_desc, 
    cdd.high_opioid_usage_flag,
    cdd.polypharmacy_status_flag, 
    cdd.frequent_er_flag, 
    cdd.frequent_imaging_flag, 
    cdd.frequent_inpatient_admission_flag, 
    cdd.comorbidity_flag

FROM shpretention_member_population pop
LEFT JOIN dw.medinsight.cchg_identification_bridge cib 
    ON cib.member_month_key = pop.member_month_key
LEFT JOIN dw.medinsight.cchg_identification_fact cif 
    ON cif.cchg_identification_key = cib.cchg_identification_key
LEFT JOIN dw.medinsight.cchg_detail_dim cdd 
    ON cdd.cchg_detail_key = cif.cchg_detail_key
LEFT JOIN dw.medinsight.cchg_category_dim primary_cchg 
    ON primary_cchg.cchg_category_key = cif.primary_cchg_category_key;

DROP TABLE
    IF EXISTS shpretention_member_attribution;
CREATE TEMP TABLE shpretention_member_attribution AS
SELECT
    pop.member_key,
    pop.most_recent_member_year_month,
    UPPER((pod.provider_organization_name)::text) AS provider_organization_name,
    pod.mchs_flag,
    pcp.month_date, 
    UPPER((prv.provider_full_name)::text) AS practitioner_full_name,                                
    prv.physical_longitude AS practitioner_physical_longitude, 
    prv.physical_latitude AS practitioner_physical_latitude, 
    prv.provider_type_code AS practitioner_provider_type_code, 
    prv.provider_type_desc AS practitioner_provider_type_desc, 
    cdztr.primary_region AS practitioner_primary_region, 
    UPPER((prc.provider_full_name)::text) AS practice_full_name, 
    spc.specialty_desc AS practitioner_primary_specialty_desc
FROM dw.member.member_pcp_month_fact pcp 
JOIN dw.provider.provider_history_dim prv 
    ON prv.provider_history_key = pcp.practitioner_provider_history_key
JOIN dw.reference.specialty_dim spc 
    ON spc.specialty_key = prv.primary_specialty_key
JOIN dw.provider.provider_history_dim prc 
    ON prc.provider_history_key = pcp.practice_provider_history_key
JOIN dw.provider.provider_organization_dim pod 
    ON pod.provider_organization_key = pcp.provider_organization_key 
JOIN shpretention_member_population pop
    ON pop.member_key = pcp.member_key
    AND pcp.month_date BETWEEN pop.from_date AND pop.thru_date 
    AND SUBSTRING(pcp.month_date_key, 1, 6) = pop.most_recent_member_year_month
LEFT JOIN dw.sandbox.care_delivery_zip_to_region cdztr --CUSTOM TABLE: ask for table owner and refresh frequency
    ON cdztr.zip_code_name = prv.physical_zip_code;


DROP TABLE IF EXISTS shpretention_crm;
CREATE TEMP TABLE shpretention_crm AS
SELECT ccf.member_key,
     COUNT(ccf.member_key) as contact_count 
   FROM member.member_crm_case_fact ccf 
   JOIN shpretention_member_population pop ON pop.member_key = ccf.member_key 
   JOIN member.member_crm_case_history_bridge crmhb
      ON crmhb.member_crm_case_key = ccf.member_crm_case_key
   JOIN member.member_crm_case_detail_dim crm
      ON ccf.member_crm_case_detail_key = crm.member_crm_case_detail_key 
WHERE CAST(ccf.create_date_time as DATE) BETWEEN  ${window_start}$ and ${window_stop}$  
GROUP BY ccf.member_key
;

DROP TABLE IF EXISTS shpretention_denied;
CREATE TEMP TABLE shpretention_denied AS
SELECT pop.shp_member_id,
      COUNT(DISTINCT(clf.claim_number)) as denied_count,
      SUM(clf.billed_amt) as denied_amt
FROM claim.claim_line_fact clf
JOIN claim.claim_detail_dim cdd
    on clf.claim_detail_key = cdd.claim_detail_key
JOIN member.member_history_dim md
    on clf.member_history_key = md.member_history_key
JOIN shpretention_member_population pop
    on pop.shp_member_id = md.shp_member_id
WHERE cdd.claim_status_code = 2
    and clf.service_thru_date between DATEADD(MONTHS, -13, ${window_start}$)  and DATEADD(MONTHS, -3, ${window_stop}$)
    and cdd.claim_type_code in (1,3,5) -- 1 = Professional, 2 = Institutional, 5 = Pharmacy 
GROUP BY pop.shp_member_id
;

DROP TABLE IF EXISTS shpretention_paid;
CREATE TEMP TABLE shpretention_paid AS
SELECT pop.shp_member_id,
      COUNT(DISTINCT(clf.claim_number)) as paid_count,
      SUM(clf.member_denial_amt) as member_amt, 
      SUM(clf.net_allowed_amt) as allowed_amt
      
FROM claim.claim_line_fact clf
JOIN claim.claim_detail_dim cdd
    on clf.claim_detail_key = cdd.claim_detail_key
JOIN member.member_history_dim md
    on clf.member_history_key = md.member_history_key
JOIN shpretention_member_population pop
    on pop.shp_member_id = md.shp_member_id
WHERE cdd.claim_status_code = 4
    and clf.service_thru_date between DATEADD(MONTHS, -13, ${window_start}$) and DATEADD(MONTHS, -3, ${window_stop}$)
    and cdd.claim_type_code in (1,3,5) -- 1 = Professional, 2 = Institutional, 5 = Pharmacy
GROUP BY pop.shp_member_id
;

DROP TABLE IF EXISTS shpretention_features;
CREATE TEMP TABLE shpretention_features AS
SELECT DISTINCT
    pop.shp_member_id,
    pop.product_type_code,
    pop.product_name,
    pop.age,
    pop.gender_code,
    pop.member_physical_zip_code,
    pop.county_name,
    pop.state_code,
    pop.rpl_theme1,
    pop.rpl_theme2,
    pop.rpl_theme3,
    pop.rpl_theme4,
    pop.rpl_themes,
    pop.member_esrd_flag,
    pop.part_c_flag,
    pop.part_d_flag,
    mara.cc_disease_cnt,
    mara.cchg_cat_rollup_code,
    mara.cchg_cat,
    mara.cancer_severity_code,
    mara.copd_severity_code,
    mara.diabetes_severity_code,
    mara.hypertension_severity_code,
    mara.high_opioid_usage_flag,
    mara.polypharmacy_status_flag,
    mara.frequent_er_flag,
    mara.frequent_imaging_flag,
    mara.frequent_inpatient_admission_flag,
    mara.comorbidity_flag,
    attr.provider_organization_name,
    attr.practitioner_full_name,
    attr.practitioner_provider_type_code,
    attr.practitioner_primary_region,
    CASE WHEN crm.contact_count IS NULL THEN 0
        ELSE crm.contact_count END as contact_count,
    d.denied_count,
    d.denied_amt,    
    p.paid_count,
    p.member_amt,
    p.allowed_amt,
    ST_DistanceSphere(ST_Point(attr.practitioner_physical_longitude, attr.practitioner_physical_latitude),
                  ST_Point(pop.member_physical_longitude, pop.member_physical_latitude)) / 1000 AS distance_in_km,
    qp.quit_plan
    
FROM shpretention_member_population pop
LEFT JOIN shpretention_member_cchg_mara mara
    ON pop.member_key = mara.member_key
LEFT JOIN shpretention_member_attribution attr
    ON pop.member_key = attr.member_key
LEFT JOIN shpretention_crm crm
    ON pop.member_key = crm.member_key
LEFT JOIN shpretention_quit_plan qp
    ON qp.shp_member_id = pop.shp_member_id    
LEFT JOIN shpretention_denied d
    ON d.shp_member_id = pop.shp_member_id
LEFT JOIN shpretention_paid p
    ON p.shp_member_id = pop.shp_member_id    


