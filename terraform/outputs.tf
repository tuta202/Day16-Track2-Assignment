output "bastion_public_ip" {
  value = aws_instance.bastion.public_ip
}

output "private_key" {
  value     = tls_private_key.lab_key.private_key_pem
  sensitive = true
}

output "alb_dns_name" {
  value = aws_lb.ai_alb.dns_name
  description = "The DNS name of the ALB to access the inference endpoint"
}

output "endpoint_url" {
  value = "http://${aws_lb.ai_alb.dns_name}/predict"
}

output "gpu_private_ip" {
  value = aws_instance.gpu_node.private_ip
}