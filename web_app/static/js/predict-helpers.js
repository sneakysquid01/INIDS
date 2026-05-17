function fillNormal() {
    document.getElementsByName('duration')[0].value = '0';
    document.getElementsByName('src_bytes')[0].value = '181';
    document.getElementsByName('dst_bytes')[0].value = '5450';
    document.getElementsByName('count')[0].value = '8';
    document.getElementsByName('srv_count')[0].value = '8';
    document.getElementsByName('serror_rate')[0].value = '0';
    document.getElementsByName('same_srv_rate')[0].value = '1';
}

function fillSuspicious() {
    document.getElementsByName('duration')[0].value = '0';
    document.getElementsByName('src_bytes')[0].value = '0';
    document.getElementsByName('dst_bytes')[0].value = '0';
    document.getElementsByName('count')[0].value = '123';
    document.getElementsByName('srv_count')[0].value = '6';
    document.getElementsByName('serror_rate')[0].value = '1';
    document.getElementsByName('same_srv_rate')[0].value = '0.05';
}

function fillHighLoad() {
    document.getElementsByName('duration')[0].value = '5';
    document.getElementsByName('src_bytes')[0].value = '1500';
    document.getElementsByName('dst_bytes')[0].value = '3200';
    document.getElementsByName('count')[0].value = '15';
    document.getElementsByName('srv_count')[0].value = '12';
    document.getElementsByName('serror_rate')[0].value = '0.1';
    document.getElementsByName('same_srv_rate')[0].value = '0.8';
}
