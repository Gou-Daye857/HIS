-- 1. 清空旧数据（防止多次运行导致数据重复或主键冲突）
DELETE FROM prescription_items;
DELETE FROM prescriptions;
DELETE FROM dispense_records;
DELETE FROM payments;
DELETE FROM visits;
DELETE FROM registrations;
DELETE FROM patients;
DELETE FROM drugs;

-- 重置所有表的自增 ID，让新数据从 ID 1 开始
DELETE FROM sqlite_sequence;

-- 2. 插入模拟药品数据 (为处方模块提供支持)
INSERT INTO drugs (drug_name, spec, unit, price, stock) VALUES
('阿莫西林胶囊', '0.25g*50粒', '盒', 15.50, 200),
('布洛芬缓释胶囊', '0.3g*24粒', '盒', 22.00, 150),
('维C银翘片', '12片*2板', '盒', 12.80, 300),
('复方甘草口服溶液', '10ml*6支', '盒', 18.50, 100),
('葡萄糖注射液', '250ml', '瓶', 5.00, 500);

-- 3. 插入模拟患者数据 (为患者管理模块提供支持)
INSERT INTO patients (name, gender, age, id_card, phone, allergy_history) VALUES
('张三', '男', 45, '11010519780101234X', '13800138000', '青霉素'),
('李四', '女', 32, '310104199105126789', '13900139000', '无'),
('王五', '男', 68, '440106195511223344', '13700137000', '海鲜'),
('赵六', '女', 25, '320102199807081122', '13600136000', '花粉'),
('孙七', '男', 12, '130103201109095566', '13500135000', '无');

-- 4. 插入模拟挂号数据 (假设关联刚刚插入的患者 ID 1到5)
INSERT INTO registrations (patient_id, dept_name, doctor_name, reg_type, queue_num, status) VALUES
(1, '内科', '李医生', '普通门诊', 1, '待诊'),
(2, '外科', '王医生', '专家门诊', 2, '待诊'),
(3, '急诊科', '张医生', '急诊', 1, '待诊'),
(4, '儿科', '陈医生', '普通门诊', 3, '就诊中'),
(5, '内科', '李医生', '普通门诊', 4, '已就诊');