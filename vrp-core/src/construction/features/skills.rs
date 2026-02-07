//! A job-vehicle skills feature.

#[cfg(test)]
#[path = "../../../tests/unit/construction/features/skills_test.rs"]
mod skills_test;

use super::*;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

custom_dimension!(pub JobSkills typeof JobSkills);
custom_dimension!(pub VehicleSkills typeof HashSet<String>);
custom_dimension!(pub JobSkillsBitset typeof JobSkillsBitset);
custom_dimension!(pub VehicleSkillsBitset typeof VehicleSkillsBitset);

/// A job skills limitation for a vehicle.
pub struct JobSkills {
    /// Vehicle should have all of these skills defined.
    pub all_of: Option<HashSet<String>>,
    /// Vehicle should have at least one of these skills defined.
    pub one_of: Option<HashSet<String>>,
    /// Vehicle should have none of these skills defined.
    pub none_of: Option<HashSet<String>>,
}

/// A bitset-based job skills representation.
#[derive(Clone, Debug)]
pub struct JobSkillsBitset {
    /// Bits for `all_of` skill constraints.
    pub all_of: Vec<u64>,
    /// Bits for `one_of` skill constraints.
    pub one_of: Vec<u64>,
    /// Bits for `none_of` skill constraints.
    pub none_of: Vec<u64>,
}

/// A bitset-based vehicle skills representation.
#[derive(Clone, Debug)]
pub struct VehicleSkillsBitset {
    /// Bits for vehicle skills.
    pub bits: Vec<u64>,
}

pub(crate) struct SkillBitsetStats {
    pub skill_count: usize,
    pub bitset_len: usize,
    pub job_count: usize,
    pub vehicle_count: usize,
}

impl JobSkills {
    /// Creates a new instance of [`JobSkills`].
    pub fn new(all_of: Option<Vec<String>>, one_of: Option<Vec<String>>, none_of: Option<Vec<String>>) -> Self {
        let map: fn(Option<Vec<_>>) -> Option<HashSet<_>> =
            |skills| skills.and_then(|v| if v.is_empty() { None } else { Some(v.into_iter().collect()) });

        Self { all_of: map(all_of), one_of: map(one_of), none_of: map(none_of) }
    }
}

pub(crate) fn apply_skill_bitsets(jobs: &mut [Job], vehicles: &mut [Vehicle]) -> Option<SkillBitsetStats> {
    let mut skill_index: HashMap<String, usize> = HashMap::new();
    let mut insert_skill = |skill: &str| {
        let next = skill_index.len();
        skill_index.entry(skill.to_string()).or_insert(next);
    };
    let mut job_count = 0;
    let mut vehicle_count = 0;

    for vehicle in vehicles.iter() {
        if let Some(skills) = vehicle.dimens.get_vehicle_skills() {
            if !skills.is_empty() {
                vehicle_count += 1;
            }
            for skill in skills {
                insert_skill(skill);
            }
        }
    }

    for job in jobs.iter() {
        if let Some(job_skills) = job.dimens().get_job_skills() {
            let mut has_any = false;
            if let Some(skills) = job_skills.all_of.as_ref() {
                if !skills.is_empty() {
                    has_any = true;
                }
                for skill in skills {
                    insert_skill(skill);
                }
            }
            if let Some(skills) = job_skills.one_of.as_ref() {
                if !skills.is_empty() {
                    has_any = true;
                }
                for skill in skills {
                    insert_skill(skill);
                }
            }
            if let Some(skills) = job_skills.none_of.as_ref() {
                if !skills.is_empty() {
                    has_any = true;
                }
                for skill in skills {
                    insert_skill(skill);
                }
            }
            if has_any {
                job_count += 1;
            }
        }
    }

    if skill_index.is_empty() {
        return None;
    }

    let bits_len = (skill_index.len() + 63) / 64;

    for vehicle in vehicles.iter_mut() {
        let mut bits = vec![0u64; bits_len];
        if let Some(skills) = vehicle.dimens.get_vehicle_skills() {
            for skill in skills {
                if let Some(&idx) = skill_index.get(skill) {
                    bits[idx / 64] |= 1u64 << (idx % 64);
                }
            }
        }
        vehicle.dimens.set_vehicle_skills_bitset(VehicleSkillsBitset { bits });
    }

    for job in jobs.iter_mut() {
        let Some(job_skills) = job.dimens().get_job_skills() else {
            continue;
        };
        let has_any = job_skills.all_of.as_ref().map_or(false, |s| !s.is_empty())
            || job_skills.one_of.as_ref().map_or(false, |s| !s.is_empty())
            || job_skills.none_of.as_ref().map_or(false, |s| !s.is_empty());
        if !has_any {
            continue;
        }

        let all_of = build_bits(job_skills.all_of.as_ref(), &skill_index, bits_len);
        let one_of = build_bits(job_skills.one_of.as_ref(), &skill_index, bits_len);
        let none_of = build_bits(job_skills.none_of.as_ref(), &skill_index, bits_len);
        let bitset = JobSkillsBitset { all_of, one_of, none_of };

        match job {
            Job::Single(single) => {
                if let Some(single) = Arc::get_mut(single) {
                    single.dimens.set_job_skills_bitset(bitset);
                }
            }
            Job::Multi(multi) => {
                if let Some(multi) = Arc::get_mut(multi) {
                    multi.dimens.set_job_skills_bitset(bitset);
                }
            }
        }
    }

    Some(SkillBitsetStats {
        skill_count: skill_index.len(),
        bitset_len: bits_len,
        job_count,
        vehicle_count,
    })
}

/// Creates a skills feature as hard constraint.
pub fn create_skills_feature(name: &str, code: ViolationCode) -> Result<Feature, GenericError> {
    FeatureBuilder::default().with_name(name).with_constraint(SkillsConstraint { code }).build()
}

struct SkillsConstraint {
    code: ViolationCode,
}

impl FeatureConstraint for SkillsConstraint {
    fn evaluate(&self, move_ctx: &MoveContext<'_>) -> Option<ConstraintViolation> {
        match move_ctx {
            MoveContext::Route { route_ctx, job, .. } => {
                if let (Some(job_bits), Some(vehicle_bits)) = (
                    job.dimens().get_job_skills_bitset(),
                    route_ctx.route().actor.vehicle.dimens.get_vehicle_skills_bitset(),
                ) {
                    let is_ok = check_all_of_bits(job_bits, vehicle_bits)
                        && check_one_of_bits(job_bits, vehicle_bits)
                        && check_none_of_bits(job_bits, vehicle_bits);
                    if !is_ok {
                        return ConstraintViolation::fail(self.code);
                    }
                } else if let Some(job_skills) = job.dimens().get_job_skills() {
                    let vehicle_skills = route_ctx.route().actor.vehicle.dimens.get_vehicle_skills();
                    let is_ok = check_all_of(job_skills, &vehicle_skills)
                        && check_one_of(job_skills, &vehicle_skills)
                        && check_none_of(job_skills, &vehicle_skills);
                    if !is_ok {
                        return ConstraintViolation::fail(self.code);
                    }
                }

                None
            }
            MoveContext::Activity { .. } => None,
        }
    }

    fn merge(&self, source: Job, candidate: Job) -> Result<Job, ViolationCode> {
        let source_skills = source.dimens().get_job_skills();
        let candidate_skills = candidate.dimens().get_job_skills();

        let check_skill_sets = |source_set: Option<&HashSet<String>>, candidate_set: Option<&HashSet<String>>| match (
            source_set,
            candidate_set,
        ) {
            (Some(_), None) | (None, None) => true,
            (None, Some(_)) => false,
            (Some(source_skills), Some(candidate_skills)) => candidate_skills.is_subset(source_skills),
        };

        let has_comparable_skills = match (source_skills, candidate_skills) {
            (Some(_), None) | (None, None) => true,
            (None, Some(_)) => false,
            (Some(source_skills), Some(candidate_skills)) => {
                check_skill_sets(source_skills.all_of.as_ref(), candidate_skills.all_of.as_ref())
                    && check_skill_sets(source_skills.one_of.as_ref(), candidate_skills.one_of.as_ref())
                    && check_skill_sets(source_skills.none_of.as_ref(), candidate_skills.none_of.as_ref())
            }
        };

        if has_comparable_skills { Ok(source) } else { Err(self.code) }
    }
}

fn check_all_of(job_skills: &JobSkills, vehicle_skills: &Option<&HashSet<String>>) -> bool {
    match (job_skills.all_of.as_ref(), vehicle_skills) {
        (Some(job_skills), Some(vehicle_skills)) => job_skills.is_subset(vehicle_skills),
        (Some(skills), None) if skills.is_empty() => true,
        (Some(_), None) => false,
        _ => true,
    }
}

fn check_one_of(job_skills: &JobSkills, vehicle_skills: &Option<&HashSet<String>>) -> bool {
    match (job_skills.one_of.as_ref(), vehicle_skills) {
        (Some(job_skills), Some(vehicle_skills)) => job_skills.iter().any(|skill| vehicle_skills.contains(skill)),
        (Some(skills), None) if skills.is_empty() => true,
        (Some(_), None) => false,
        _ => true,
    }
}

fn check_none_of(job_skills: &JobSkills, vehicle_skills: &Option<&HashSet<String>>) -> bool {
    match (job_skills.none_of.as_ref(), vehicle_skills) {
        (Some(job_skills), Some(vehicle_skills)) => job_skills.is_disjoint(vehicle_skills),
        _ => true,
    }
}

fn check_all_of_bits(job_bits: &JobSkillsBitset, vehicle_bits: &VehicleSkillsBitset) -> bool {
    if job_bits.all_of.is_empty() {
        return true;
    }
    if job_bits.all_of.len() != vehicle_bits.bits.len() {
        return false;
    }

    for (job, vehicle) in job_bits.all_of.iter().zip(vehicle_bits.bits.iter()) {
        if job & !vehicle != 0 {
            return false;
        }
    }
    true
}

fn check_one_of_bits(job_bits: &JobSkillsBitset, vehicle_bits: &VehicleSkillsBitset) -> bool {
    if job_bits.one_of.is_empty() {
        return true;
    }
    if job_bits.one_of.len() != vehicle_bits.bits.len() {
        return false;
    }

    let mut has_any = false;
    for (job, vehicle) in job_bits.one_of.iter().zip(vehicle_bits.bits.iter()) {
        if *job != 0 {
            has_any = true;
            if job & vehicle != 0 {
                return true;
            }
        }
    }

    !has_any
}

fn check_none_of_bits(job_bits: &JobSkillsBitset, vehicle_bits: &VehicleSkillsBitset) -> bool {
    if job_bits.none_of.is_empty() {
        return true;
    }
    if job_bits.none_of.len() != vehicle_bits.bits.len() {
        return false;
    }

    for (job, vehicle) in job_bits.none_of.iter().zip(vehicle_bits.bits.iter()) {
        if job & vehicle != 0 {
            return false;
        }
    }
    true
}

fn build_bits(
    skills: Option<&HashSet<String>>,
    skill_index: &HashMap<String, usize>,
    bits_len: usize,
) -> Vec<u64> {
    match skills {
        Some(skills) if !skills.is_empty() => {
            let mut bits = vec![0u64; bits_len];
            for skill in skills {
                if let Some(&idx) = skill_index.get(skill) {
                    bits[idx / 64] |= 1u64 << (idx % 64);
                }
            }
            bits
        }
        _ => Vec::new(),
    }
}
