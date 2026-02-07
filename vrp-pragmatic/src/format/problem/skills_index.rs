use super::{ApiProblem, JobSkills as ApiJobSkills};
use std::collections::{BTreeSet, HashMap};
use vrp_core::construction::features::{JobSkillsBitset, VehicleSkillsBitset};

pub(super) struct SkillIndex {
    index: HashMap<String, usize>,
    bits_len: usize,
}

impl SkillIndex {
    pub(super) fn new(api_problem: &ApiProblem) -> Option<Self> {
        let mut all = BTreeSet::<String>::new();

        for job in api_problem.plan.jobs.iter() {
            if let Some(skills) = job.skills.as_ref() {
                add_skills(&mut all, skills.all_of.as_ref());
                add_skills(&mut all, skills.one_of.as_ref());
                add_skills(&mut all, skills.none_of.as_ref());
            }
        }

        for vehicle in api_problem.fleet.vehicles.iter() {
            add_skills(&mut all, vehicle.skills.as_ref());
        }

        if all.is_empty() {
            return None;
        }

        let skills = all.into_iter().collect::<Vec<_>>();
        let bits_len = (skills.len() + 63) / 64;
        let index = skills.into_iter().enumerate().map(|(idx, skill)| (skill, idx)).collect();

        Some(Self { index, bits_len })
    }

    pub(super) fn make_job_bitset(&self, skills: &Option<ApiJobSkills>) -> Option<JobSkillsBitset> {
        let skills = skills.as_ref()?;

        let all_of = self.build_bits(skills.all_of.as_ref());
        let one_of = self.build_bits(skills.one_of.as_ref());
        let none_of = self.build_bits(skills.none_of.as_ref());

        if all_of.is_empty() && one_of.is_empty() && none_of.is_empty() {
            None
        } else {
            Some(JobSkillsBitset { all_of, one_of, none_of })
        }
    }

    pub(super) fn make_vehicle_bitset(&self, skills: &Option<Vec<String>>) -> Option<VehicleSkillsBitset> {
        let bits = self.build_bits(skills.as_ref());
        if bits.is_empty() { None } else { Some(VehicleSkillsBitset { bits }) }
    }

    fn build_bits(&self, skills: Option<&Vec<String>>) -> Vec<u64> {
        match skills {
            Some(skills) if !skills.is_empty() => {
                let mut bits = vec![0u64; self.bits_len];
                for skill in skills {
                    if let Some(&idx) = self.index.get(skill) {
                        bits[idx / 64] |= 1u64 << (idx % 64);
                    }
                }
                bits
            }
            _ => Vec::new(),
        }
    }
}

fn add_skills(set: &mut BTreeSet<String>, skills: Option<&Vec<String>>) {
    if let Some(skills) = skills {
        for skill in skills {
            set.insert(skill.clone());
        }
    }
}
